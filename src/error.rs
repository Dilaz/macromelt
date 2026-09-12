//! Error types returned by the parsing and export stages.

/// Failure while parsing an XMED byte stream into an [`XmedFile`](crate::XmedFile).
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// The bytes are not an XMED stream: the leading `XMED` magic or the
    /// Director header that follows it did not match.
    #[error("not an XMED stream: {0}")]
    Header(String),
    /// A chunk failed to parse. `offset` is the byte offset into the stream at
    /// which the failing chunk started.
    #[error("chunk parse failed at byte {offset}: {reason}")]
    Chunk {
        /// Byte offset of the failing chunk from the start of the stream.
        offset: usize,
        /// What the chunk parser rejected.
        reason: String,
    },
}

/// Failure while baking a parsed [`XmedFile`](crate::XmedFile) to glTF.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// A geometry stream ended before the mesh declaration's promised element
    /// count was reached, so the mesh would export truncated. The bake refuses
    /// rather than write a partial model.
    #[error("geometry \"{mesh}\" decoded {got} of {declared} {what}")]
    IncompleteDecode {
        /// Name of the Director mesh whose geometry is short.
        mesh: String,
        /// Which element ran out: `"positions"` or `"faces"`.
        what: &'static str,
        /// How many elements the arithmetic decoder produced.
        got: usize,
        /// How many the 0x45 mesh declaration promised.
        declared: usize,
    },
    /// Any other export failure, with a human-readable description.
    #[error("{0}")]
    Other(String),
}
