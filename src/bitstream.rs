//! IFX/U3D-style arithmetic bitstream decoder.
//!
//! Ported from Intel's U3D SDK (Apache 2.0) CIFXBitStreamX.
//! The core arithmetic coding algorithm is shared between IFX v2 and U3D.

use log::{Level, log_enabled, trace};
use std::collections::HashMap;

const AC_STATIC_FULL: u32 = 0x0000_0400;
// Binary FUN_7a11e220: contexts >= 0x43ff fall through to a raw ReadU32X.
const AC_MAX_RANGE: u32 = 0x0000_43ff;
const HALF_MASK: u32 = 0x8000_8000;
const NOT_HALF_MASK: u32 = 0x7FFF_7FFF;
const QUARTER_MASK: u32 = 0x4000_4000;
const NOT_THREE_QUARTER_MASK: u32 = 0x3FFF_3FFF;

const SWAP8: [u32; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
const READ_COUNT: [u32; 16] = [4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];
const FAST_NOT_MASK: [u32; 5] = [
    0xFFFF_FFFF,
    0x7FFF_7FFF,
    0x3FFF_3FFF,
    0x1FFF_1FFF,
    0x0FFF_0FFF,
];

/// Dynamic histogram using dense array indexed by symbol value.
/// Matches the C++ IFXHistogramDynamic: symbols are ordered by numeric value,
/// not insertion order. Includes elephant scaling (halve frequencies at 0x1FFF).
#[derive(Clone, Debug)]
struct DynamicHistogram {
    /// Symbol count array — indexed by symbol value. Length = max_symbol + 1.
    counts: Vec<u32>,
    total: u32,
}

const ELEPHANT_THRESHOLD: u32 = 0x1FFF;

impl DynamicHistogram {
    fn new() -> Self {
        // Initial state: only escape symbol (0) with frequency 1
        Self {
            counts: vec![1],
            total: 1,
        }
    }

    fn get_symbol_from_freq(&self, cum_freq: u32) -> u32 {
        let mut cumulative = 0u32;
        for (symbol, &freq) in self.counts.iter().enumerate() {
            if freq == 0 {
                continue;
            }
            cumulative += freq;
            if cumulative > cum_freq {
                return symbol as u32;
            }
        }
        // Fallback: last non-zero symbol
        for (symbol, &freq) in self.counts.iter().enumerate().rev() {
            if freq > 0 {
                return symbol as u32;
            }
        }
        0
    }

    fn get_cum_freq(&self, symbol: u32) -> u32 {
        let idx = symbol as usize;
        let mut cum = 0u32;
        for i in 0..idx.min(self.counts.len()) {
            cum += self.counts[i];
        }
        cum
    }

    fn get_freq(&self, symbol: u32) -> u32 {
        let idx = symbol as usize;
        if idx < self.counts.len() {
            self.counts[idx]
        } else {
            0
        }
    }

    fn add_symbol(&mut self, symbol: u32) {
        let idx = symbol as usize;
        // `IFXHistogramDynamic::AddSymbol` ignores symbols above
        // `m_uMaximumSymbolInHistogram = 0xFFFF` (IFXHistogramDynamic.cpp:29,355)
        // — no count, no elephant rescale. Large raw escapes hit this routinely.
        if idx > 0xFFFF {
            return;
        }
        // Grow the array if needed
        if idx >= self.counts.len() {
            self.counts.resize(idx + 1, 0);
        }
        self.counts[idx] += 1;
        self.total += 1;

        // Elephant scaling: halve all counts when total exceeds threshold
        if self.total >= ELEPHANT_THRESHOLD {
            self.total = 0;
            for count in &mut self.counts {
                *count = (*count + 1) / 2; // round up to prevent zero
                self.total += *count;
            }
            // Ensure escape symbol (0) never reaches 0
            if self.counts[0] == 0 {
                self.counts[0] = 1;
                self.total += 1;
            }
        }
    }
}

/// Reader for the arithmetic-coded IFX bitstream that carries every compressed
/// XMED payload: CLOD geometry, motion tracks and bone weights.
///
/// The stream is an array of little-endian 32-bit words read bit by bit from
/// the low end of each word. On top of it sits the U3D/IFX arithmetic coder:
/// values arrive as *symbols* drawn from a **context**, a numbered probability
/// model. Context ids at or below `AC_STATIC_FULL` (0x400) select a *dynamic*
/// context, whose histogram this reader adapts as symbols arrive — the same
/// symbol grows cheaper the more often the encoder used it. Ids above it select
/// a *static* context, a uniform model over `context - AC_STATIC_FULL` equally
/// likely symbols that carries no history, which is what the byte-wise literal
/// path uses. Ids at or above `AC_MAX_RANGE` are outside the coder's range and
/// fall through to plain reads.
///
/// Inside a compressed read, symbol 0 is the **escape symbol**: the value is
/// not in the context's histogram yet, so the literal follows through the
/// uniform static byte path and is then added to the histogram. Any other
/// symbol `s` decodes to the value `s - 1`.
pub(crate) struct BitStream {
    /// Stream payload as little-endian 32-bit words, zero-padded to a word.
    words: Vec<u32>,
    /// Index of the word the cursor is in.
    data_pos: usize,
    /// Bit cursor inside that word, `0..32`, counted from its low bit.
    bit_offset: u32,
    /// Cached `words[data_pos]`.
    data_local: u32,
    /// Cached `words[data_pos + 1]`, for reads straddling a word boundary.
    data_local_next: u32,

    // Arithmetic coder state
    /// Top of the coder's current 16-bit interval.
    ac_high: u32,
    /// Bottom of the coder's current 16-bit interval.
    ac_low: u32,
    /// The 16-bit code word read from the stream for the symbol in flight.
    ac_code: u32,
    /// Bits the coder owes the cursor, consumed at the next renormalization.
    ac_underflow: u32,

    // Dynamic contexts
    /// Adaptive histograms, keyed by context id (ids `<= AC_STATIC_FULL`).
    contexts: HashMap<u32, DynamicHistogram>,

    // Debug tracing
    /// Emit one `macromelt::bittrace` line per compressed read.
    pub(crate) trace_reads: bool,
    /// Caller identification prefixed to every `trace_reads` line.
    pub(crate) trace_label: String,
    /// Emit the compact full trace — format `#n TYPE ctx=X val=Y`, matching the
    /// winedbg ground-truth capture of the original decoder.
    pub(crate) full_trace: bool,
    /// Sequence number of the next full-trace line.
    pub(crate) read_idx: u32,
}

impl BitStream {
    /// Wrap a compressed payload. The data is zero-padded to a whole number of
    /// 32-bit words and the arithmetic coder starts in its pristine state
    /// (`low = 0`, `high = 0xFFFF`, no underflow, no contexts).
    pub(crate) fn new(data: &[u8]) -> Self {
        let mut padded = data.to_vec();
        while padded.len() % 4 != 0 {
            padded.push(0);
        }
        let words: Vec<u32> = padded
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let local = words.first().copied().unwrap_or(0);
        let local_next = words.get(1).copied().unwrap_or(0);

        Self {
            words,
            data_pos: 0,
            bit_offset: 0,
            data_local: local,
            data_local_next: local_next,
            ac_high: 0x0000_FFFF,
            ac_low: 0x0000_0000,
            ac_code: 0,
            ac_underflow: 0,
            contexts: HashMap::new(),
            trace_reads: false,
            trace_label: String::new(),
            full_trace: false,
            read_idx: 0,
        }
    }

    #[inline]
    fn ft(&mut self, ty: &str, ctx: u32, val: u32) {
        if self.full_trace {
            // post-read AC state, to localize internal-state divergence vs the binary
            trace!(
                target: "macromelt::bittrace",
                "#{} {} ctx={} val={} | low={:#06x} high={:#06x} uf={}",
                self.read_idx, ty, ctx, val, self.ac_low, self.ac_high, self.ac_underflow
            );
            self.read_idx = self.read_idx.wrapping_add(1);
        }
    }

    /// Cursor position, in bits from the start of the payload.
    pub(crate) fn bit_count(&self) -> u32 {
        // wrapping_* so that a desync that runs the cursor far past the buffer
        // produces garbage rather than panicking on debug-build overflow.
        (self.data_pos as u32)
            .wrapping_mul(32)
            .wrapping_add(self.bit_offset)
    }

    /// Payload length in bits, including the zero padding added by `new`.
    pub(crate) fn total_bits(&self) -> u32 {
        (self.words.len() as u32) * 32
    }

    fn increment_pos(&mut self) {
        self.data_pos += 1;
        self.data_local = self.words.get(self.data_pos).copied().unwrap_or(0);
        self.data_local_next = self.words.get(self.data_pos + 1).copied().unwrap_or(0);
    }

    /// Move the cursor to an absolute bit position, refreshing the word cache.
    pub(crate) fn seek_to_bit(&mut self, position: u32) {
        self.data_pos = (position >> 5) as usize;
        self.bit_offset = position & 31;
        self.data_local = self.words.get(self.data_pos).copied().unwrap_or(0);
        self.data_local_next = self.words.get(self.data_pos + 1).copied().unwrap_or(0);
    }

    /// Read one raw bit, least significant bit of the current word first.
    pub(crate) fn read_bit(&mut self) -> u32 {
        let value = (self.data_local >> self.bit_offset) & 1;
        self.bit_offset += 1;
        if self.bit_offset >= 32 {
            self.bit_offset -= 32;
            self.increment_pos();
        }
        value
    }

    fn read_15_bits(&mut self) -> u32 {
        let mut value = self.data_local >> self.bit_offset;
        if self.bit_offset > 17 {
            value |= self.data_local_next << (32 - self.bit_offset);
        }
        value = value.wrapping_add(value); // left shift by 1
        value = (SWAP8[((value >> 12) & 0xF) as usize])
            | (SWAP8[((value >> 8) & 0xF) as usize] << 4)
            | (SWAP8[((value >> 4) & 0xF) as usize] << 8)
            | (SWAP8[(value & 0xF) as usize] << 12);

        self.bit_offset += 15;
        if self.bit_offset >= 32 {
            self.bit_offset -= 32;
            self.increment_pos();
        }
        value & 0x7FFF
    }

    /// Read 8 raw bits — no arithmetic coding, the cursor simply advances.
    ///
    /// XMED uses raw reads for escape values (unlike U3D which goes through AC).
    pub(crate) fn read_u8(&mut self) -> u8 {
        let mut value = self.data_local >> self.bit_offset;
        if self.bit_offset > 24 {
            value |= self.data_local_next << (32 - self.bit_offset);
        }
        self.bit_offset += 8;
        if self.bit_offset >= 32 {
            self.bit_offset -= 32;
            self.increment_pos();
        }
        (value & 0xFF) as u8
    }

    /// Read 16 raw bits, little-endian, no arithmetic coding.
    pub(crate) fn read_u16(&mut self) -> u16 {
        let mut value = self.data_local >> self.bit_offset;
        if self.bit_offset > 16 {
            value |= self.data_local_next << (32 - self.bit_offset);
        }
        self.bit_offset += 16;
        if self.bit_offset >= 32 {
            self.bit_offset -= 32;
            self.increment_pos();
        }
        (value & 0xFFFF) as u16
    }

    /// Read 32 raw bits, little-endian, no arithmetic coding.
    pub(crate) fn read_u32(&mut self) -> u32 {
        let mut value = self.data_local >> self.bit_offset;
        if self.bit_offset > 0 {
            value |= self.data_local_next << (32 - self.bit_offset);
        }
        self.bit_offset += 32;
        if self.bit_offset >= 32 {
            self.bit_offset -= 32;
            self.increment_pos();
        }
        value
    }

    /// Read 32 raw bits and reinterpret them as an IEEE-754 `f32`.
    pub(crate) fn read_f32(&mut self) -> f32 {
        f32::from_bits(self.read_u32())
    }

    // ── Arithmetic coding ───────────────────────────────────────

    fn read_symbol_static(&mut self, context: u32) -> u32 {
        let position = self.bit_count();

        self.ac_code = self.read_bit();
        self.bit_offset += self.ac_underflow;
        while self.bit_offset >= 32 {
            self.bit_offset -= 32;
            self.increment_pos();
        }

        let temp = self.read_15_bits();
        self.ac_code = (self.ac_code << 15) | temp;
        self.seek_to_bit(position);

        let num_symbols = context - AC_STATIC_FULL;
        let total_cum_freq = num_symbols;
        let ac_range = self.ac_high.wrapping_add(1).wrapping_sub(self.ac_low);

        let code_cum_freq = (total_cum_freq
            .wrapping_mul(1u32.wrapping_add(self.ac_code).wrapping_sub(self.ac_low))
            .wrapping_sub(1))
            / ac_range;
        // NOTE: the binary's static reader (FUN_7a11cf90) returns `codeCumFreq`
        // directly (`*param_2 = uVar8` where uVar8 = codeCumFreq), NOT
        // `codeCumFreq + 1` as the U3D reference does. The AC *state* update is
        // identical either way (high/low use codeCumFreq+1 / codeCumFreq), so this
        // only changes the returned symbol — but that difference is load-bearing:
        // `read_compressed_u32` tests `symbol != 0` to decide escape, so the binary
        // ESCAPES when codeCumFreq==0 on a static context, while a +1 convention
        // never would. Match the binary: return codeCumFreq, freq window stays
        // [codeCumFreq, codeCumFreq+1).
        let value = code_cum_freq;

        let value_freq = 1u32;
        let value_cum_freq = code_cum_freq;

        let high = self
            .ac_low
            .wrapping_sub(1)
            .wrapping_add(ac_range.wrapping_mul(value_cum_freq + value_freq) / total_cum_freq);
        let low = self.ac_low + ac_range.wrapping_mul(value_cum_freq) / total_cum_freq;

        let mut state = (low << 16) | high;
        let mut bit_count = READ_COUNT[(((low >> 12) ^ (high >> 12)) & 0xF) as usize];
        state &= FAST_NOT_MASK[bit_count as usize];
        state <<= bit_count;
        state |= (1u32 << bit_count) - 1;

        let bc2 = READ_COUNT[(((state >> 12) ^ (state >> 28)) & 0xF) as usize];
        state &= FAST_NOT_MASK[bc2 as usize];
        state <<= bc2;
        bit_count += bc2;
        state |= (1u32 << bc2) - 1;

        let mut masked = HALF_MASK & state;
        let mut half_guard = 0u32;
        while masked == 0 || masked == HALF_MASK {
            state = ((NOT_HALF_MASK & state) << 1) | 1;
            masked = HALF_MASK & state;
            bit_count += 1;
            half_guard += 1;
            if half_guard > 64 {
                panic!(
                    "AC HALF loop exceeded 64 iterations — desync at bit {}",
                    self.bit_count()
                );
            }
        }

        let saved_bits = masked;
        if bit_count > 0 {
            bit_count += self.ac_underflow;
            self.ac_underflow = 0;
        }

        masked = QUARTER_MASK & state;
        let mut underflow = 0u32;
        while masked == 0x4000_0000 {
            state &= NOT_THREE_QUARTER_MASK;
            state = state.wrapping_add(state);
            state |= 1;
            masked = QUARTER_MASK & state;
            underflow += 1;
            if underflow > 64 {
                panic!(
                    "AC QUARTER loop exceeded 64 iterations — desync at bit {}",
                    self.bit_count()
                );
            }
        }
        self.ac_underflow += underflow;
        state |= saved_bits;
        self.ac_low = (state >> 16) & 0xFFFF;
        self.ac_high = state & 0xFFFF;

        self.bit_offset += bit_count;
        while self.bit_offset >= 32 {
            self.bit_offset -= 32;
            self.increment_pos();
        }

        value
    }

    fn read_symbol_dynamic(&mut self, context_id: u32) -> u32 {
        let ctx = self
            .contexts
            .entry(context_id)
            .or_insert_with(DynamicHistogram::new)
            .clone();

        let position = self.bit_count();
        self.ac_code = self.read_bit();
        self.bit_offset += self.ac_underflow;
        while self.bit_offset >= 32 {
            self.bit_offset -= 32;
            self.increment_pos();
        }
        let temp = self.read_15_bits();
        self.ac_code = (self.ac_code << 15) | temp;
        self.seek_to_bit(position);

        let total_cum_freq = ctx.total;
        let ac_range = self.ac_high.wrapping_add(1).wrapping_sub(self.ac_low);

        let code_cum_freq = (total_cum_freq
            .wrapping_mul(1u32.wrapping_add(self.ac_code).wrapping_sub(self.ac_low))
            .wrapping_sub(1))
            / ac_range;

        let value = ctx.get_symbol_from_freq(code_cum_freq);
        let value_cum_freq = ctx.get_cum_freq(value);
        let value_freq = ctx.get_freq(value);

        // Update context
        self.contexts
            .get_mut(&context_id)
            .unwrap()
            .add_symbol(value);

        let high = self
            .ac_low
            .wrapping_sub(1)
            .wrapping_add(ac_range.wrapping_mul(value_cum_freq + value_freq) / total_cum_freq);
        let low = self.ac_low + ac_range.wrapping_mul(value_cum_freq) / total_cum_freq;

        let mut state = (low << 16) | high;
        let mut bit_count = READ_COUNT[(((low >> 12) ^ (high >> 12)) & 0xF) as usize];
        state &= FAST_NOT_MASK[bit_count as usize];
        state <<= bit_count;
        state |= (1u32 << bit_count) - 1;

        let mut masked = HALF_MASK & state;
        let mut half_guard = 0u32;
        while masked == 0 || masked == HALF_MASK {
            state = ((NOT_HALF_MASK & state) << 1) | 1;
            masked = HALF_MASK & state;
            bit_count += 1;
            half_guard += 1;
            if half_guard > 64 {
                panic!(
                    "AC HALF loop exceeded 64 iterations — desync at bit {}",
                    self.bit_count()
                );
            }
        }

        let saved_bits = masked;
        if bit_count > 0 {
            bit_count += self.ac_underflow;
            self.ac_underflow = 0;
        }

        masked = QUARTER_MASK & state;
        let mut underflow = 0u32;
        while masked == 0x4000_0000 {
            state &= NOT_THREE_QUARTER_MASK;
            state = state.wrapping_add(state);
            state |= 1;
            masked = QUARTER_MASK & state;
            underflow += 1;
            if underflow > 64 {
                panic!(
                    "AC QUARTER loop exceeded 64 iterations — desync at bit {}",
                    self.bit_count()
                );
            }
        }
        self.ac_underflow += underflow;
        state |= saved_bits;
        self.ac_low = (state >> 16) & 0xFFFF;
        self.ac_high = state & 0xFFFF;

        self.bit_offset += bit_count;
        while self.bit_offset >= 32 {
            self.bit_offset -= 32;
            self.increment_pos();
        }

        value
    }

    /// Decode one symbol from `context`.
    ///
    /// Context 0 means "no model": it decodes through the uniform 256-symbol
    /// static context, same as a literal byte. A context above `AC_STATIC_FULL`
    /// is static — uniform over `context - AC_STATIC_FULL` symbols, no history
    /// kept. Anything else is dynamic: the symbol is drawn from, and then added
    /// to, this context's adaptive histogram.
    pub(crate) fn read_symbol(&mut self, context: u32) -> u32 {
        if context == 0 {
            self.read_symbol_static(AC_STATIC_FULL + 256)
        } else if context > AC_STATIC_FULL {
            self.read_symbol_static(context)
        } else {
            self.read_symbol_dynamic(context)
        }
    }

    /// Reverse bits within a byte (U3D SwapBits8).
    fn swap_bits_8(v: u8) -> u8 {
        let v = v as u32;
        let swapped = (SWAP8[(v & 0xF) as usize] << 4) | SWAP8[((v >> 4) & 0xF) as usize];
        swapped as u8
    }

    /// Read one byte through the uniform-256 AC static context
    /// (U3D `ReadSymbolContext8`).
    ///
    /// When the coder's state is pristine — the interval untouched and no
    /// underflow owed — the encoding degenerates to the raw 8 bits, which is
    /// the fast path. Otherwise the byte is decoded through the static context
    /// and bit-reversed.
    pub(crate) fn read_symbol_context8(&mut self) -> u8 {
        // Fast path: when AC state is pristine, just read 8 raw bits
        if self.ac_high == 0x0000_FFFF && self.ac_low == 0x0000_0000 && self.ac_underflow == 0 {
            return self.read_u8();
        }
        // Slow path: AC static decode + SwapBits8.
        // The binary's ReadU8X (FUN_7a11dbf0) does NOT subtract 1 before SwapBits8
        // — it feeds the static reader's raw return straight in. Since our static
        // reader now returns `codeCumFreq` (binary convention, == U3D's value-1),
        // there is no `-1` here either; the two cancel out to SwapBits8(codeCumFreq).
        let sym = self.read_symbol_static(AC_STATIC_FULL + 256);
        Self::swap_bits_8(sym as u8)
    }

    /// Read four `read_symbol_context8` bytes as a little-endian `u32`
    /// (U3D SDK `ReadU32X`).
    ///
    /// In pristine AC state, this is equivalent to raw read.
    /// In non-pristine state, this goes through AC static + SwapBits8.
    pub(crate) fn read_u32_via_context8(&mut self) -> u32 {
        let b0 = self.read_symbol_context8() as u32;
        let b1 = self.read_symbol_context8() as u32;
        let b2 = self.read_symbol_context8() as u32;
        let b3 = self.read_symbol_context8() as u32;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    /// Read an IEEE-754 `f32` through `read_u32_via_context8`
    /// (matches U3D SDK's `ReadF32X`).
    pub(crate) fn read_f32_via_context8(&mut self) -> f32 {
        f32::from_bits(self.read_u32_via_context8())
    }

    /// Read u16 through AC static context (two ReadSymbolContext8 calls, LE).
    fn read_u16_ac(&mut self) -> u16 {
        let lo = self.read_symbol_context8() as u16;
        let hi = self.read_symbol_context8() as u16;
        lo | (hi << 8)
    }

    /// Read u32 through AC static context (four ReadSymbolContext8 calls, LE).
    fn read_u32_ac(&mut self) -> u32 {
        let b0 = self.read_symbol_context8() as u32;
        let b1 = self.read_symbol_context8() as u32;
        let b2 = self.read_symbol_context8() as u32;
        let b3 = self.read_symbol_context8() as u32;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    /// Read a compressed `u32` from `context`.
    ///
    /// "Compressed" means the value comes out of the context's model rather
    /// than off the wire: a symbol `s != 0` decodes to `s - 1` in as few bits as
    /// the model allows, while the escape symbol 0 means the value has not been
    /// seen in this context yet, so the full 32 bits follow through the static
    /// byte path and the value is then added to the dynamic histogram. This is
    /// the widest of the three variants — `read_compressed_u16` and
    /// `read_compressed_u8` are the same protocol with a 16- and 8-bit escape
    /// literal and a correspondingly narrower return type.
    pub(crate) fn read_compressed_u32(&mut self, context: u32) -> u32 {
        if context == 0 || context >= AC_MAX_RANGE {
            // Out-of-AC-range context (the normal acos-angle 2nd magnitude, etc.):
            // the binary escapes to ReadU32X = the context8×4 AC byte path, NOT a
            // raw 32-bit read. Both consume the same 32 bits, but the byte path also
            // updates ac_low/ac_high; a raw read leaves the AC state stale and
            // silently corrupts every downstream value (verified against a winedbg
            // (ctx,value) trace of the original on arrow.xmed: ctx=22761 → 339, with
            // the stream realigning only when AC state is carried through).
            let v = self.read_u32_via_context8();
            self.ft("U32", context, v);
            return v;
        }

        let pos_before = self.bit_count();
        let symbol = self.read_symbol(context);
        if symbol != 0 {
            self.ft("U32", context, symbol - 1);
            if self.trace_reads {
                trace!(
                    target: "macromelt::bittrace",
                    "      {} read_cu32(ctx={}) @bit{}: sym={} → val={} (now @bit{}, high={:#06x} low={:#06x} uf={})",
                    self.trace_label,
                    context,
                    pos_before,
                    symbol,
                    symbol - 1,
                    self.bit_count(),
                    self.ac_high,
                    self.ac_low,
                    self.ac_underflow,
                );
            }
            symbol - 1
        } else {
            // Escape: read through AC static context (U3D ReadSymbolContext8)
            let value = self.read_u32_ac();
            if context <= AC_STATIC_FULL {
                self.contexts
                    .entry(context)
                    .or_insert_with(DynamicHistogram::new)
                    .add_symbol(value + 1);
            }
            self.ft("U32", context, value);
            if self.trace_reads {
                trace!(
                    target: "macromelt::bittrace",
                    "      {} read_cu32(ctx={}) @bit{}: ESCAPE → val={} (now @bit{}, high={:#06x} low={:#06x} uf={})",
                    self.trace_label,
                    context,
                    pos_before,
                    value,
                    self.bit_count(),
                    self.ac_high,
                    self.ac_low,
                    self.ac_underflow,
                );
            }
            value
        }
    }

    /// Read a compressed `u16` from `context` — `read_compressed_u32`'s
    /// protocol with a 16-bit escape literal. An out-of-range context reads 16
    /// raw bits here rather than taking the static byte path.
    pub(crate) fn read_compressed_u16(&mut self, context: u32) -> u16 {
        if context == 0 || context >= AC_MAX_RANGE {
            return self.read_u16();
        }

        let pos_before = self.bit_count();
        let symbol = self.read_symbol(context);
        let pos_after_sym = self.bit_count();
        if symbol != 0 {
            let val = (symbol - 1) as u16;
            if self.trace_reads {
                trace!(
                    target: "macromelt::bittrace",
                    "      {} read_cu16(ctx={}) @bit{}: sym={} → val={} (consumed {} bits, now @bit{})",
                    self.trace_label,
                    context,
                    pos_before,
                    symbol,
                    val,
                    self.bit_count() - pos_before,
                    self.bit_count()
                );
            }
            val
        } else {
            // Escape: read through AC static context (U3D ReadSymbolContext8)
            let value = self.read_u16_ac();
            if self.trace_reads {
                trace!(
                    target: "macromelt::bittrace",
                    "      {} read_cu16(ctx={}) @bit{}: ESCAPE sym_bits={} val={} (total {} bits, now @bit{})",
                    self.trace_label,
                    context,
                    pos_before,
                    pos_after_sym - pos_before,
                    value,
                    self.bit_count() - pos_before,
                    self.bit_count()
                );
            }
            if context <= AC_STATIC_FULL {
                self.contexts
                    .entry(context)
                    .or_insert_with(DynamicHistogram::new)
                    .add_symbol(value as u32 + 1);
            }
            value
        }
    }

    /// Read a compressed `u8` from `context` — `read_compressed_u32`'s protocol
    /// with a single-byte escape literal (one `read_symbol_context8`). An
    /// out-of-range context reads 8 raw bits.
    pub(crate) fn read_compressed_u8(&mut self, context: u32) -> u8 {
        if context == 0 || context >= AC_MAX_RANGE {
            let v = self.read_u8();
            self.ft("U8", context, v as u32);
            return v;
        }

        let pos_before = self.bit_count();
        let symbol = self.read_symbol(context);
        let pos_after_sym = self.bit_count();
        if symbol != 0 {
            let val = (symbol - 1) as u8;
            self.ft("U8", context, val as u32);
            if self.trace_reads {
                trace!(
                    target: "macromelt::bittrace",
                    "      {} read_cu8(ctx={}) @bit{}: sym={} → val={} (consumed {} bits, now @bit{})",
                    self.trace_label,
                    context,
                    pos_before,
                    symbol,
                    val,
                    self.bit_count() - pos_before,
                    self.bit_count()
                );
            }
            val
        } else {
            // Escape: read through AC static context (U3D ReadSymbolContext8)
            let value = self.read_symbol_context8();
            self.ft("U8", context, value as u32);
            if self.trace_reads {
                trace!(
                    target: "macromelt::bittrace",
                    "      {} read_cu8(ctx={}) @bit{}: ESCAPE sym_bits={} val={} (0x{:02X}) (total {} bits, now @bit{})",
                    self.trace_label,
                    context,
                    pos_before,
                    pos_after_sym - pos_before,
                    value,
                    value,
                    self.bit_count() - pos_before,
                    self.bit_count()
                );
            }
            if context <= AC_STATIC_FULL {
                self.contexts
                    .entry(context)
                    .or_insert_with(DynamicHistogram::new)
                    .add_symbol(value as u32 + 1);
            }
            value
        }
    }

    /// Number of underflow bits the coder still owes the cursor.
    pub(crate) fn ac_underflow(&self) -> u32 {
        self.ac_underflow
    }

    /// Trace the coder's interval, code word, underflow and every live context
    /// histogram on `macromelt::bittrace`. `label` marks the call site.
    pub(crate) fn dump_ac_state(&self, label: &str) {
        if !log_enabled!(target: "macromelt::bittrace", Level::Trace) {
            return;
        }
        trace!(
            target: "macromelt::bittrace",
            "    AC[{}]: bit={} high={:#06x} low={:#06x} code={:#06x} uf={} ctxs={}",
            label,
            self.bit_count(),
            self.ac_high,
            self.ac_low,
            self.ac_code,
            self.ac_underflow,
            self.contexts.len()
        );
        for (id, hist) in &self.contexts {
            let syms: Vec<String> = hist
                .counts
                .iter()
                .enumerate()
                .filter(|&(_, c)| *c > 0)
                .map(|(s, c)| format!("{}:{}", s, c))
                .collect();
            trace!(
                target: "macromelt::bittrace",
                "      ctx{}: total={} syms=[{}]",
                id,
                hist.total,
                syms.join(",")
            );
        }
    }
}
