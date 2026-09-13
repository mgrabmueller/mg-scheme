//! Lexer for the R6RS lexical syntax (R6RS section 4.2 / 11.7).
//!
//! Produces a flat stream of [`Token`]s together with byte span
//! information. The lexer handles whitespace, line comments (`;`),
//! datum comments (`#;`), block comments (`#| ... |#` with nesting),
//! booleans, characters, numbers, strings (with escapes), the
//! parenthesization marks, quote abbreviations (`'`, `` ` ``, `,`,
//! `,@`), the vector/bytevector prefixes `#(` and `#vu8(`, the dot `.`,
//! and symbols (which subsume everything else not otherwise claimed).
//!
//! Identifiers follow the R6RS constituent + inline-hex-escape rules
//! that are practical to implement with a byte scanner.

use crate::ast::Number;

/// A 1-based position in the source: line and column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub line: u32,
    pub column: u32,
    pub offset: usize,
}

/// A lexical token produced by [`tokenize`].
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    LeftParen,
    RightParen,
    /// The `.` used in `(a . b)`. Only emitted when it stands alone as
    /// a dotted-pair marker; `.` as part of a larger token is a symbol.
    Dot,
    Quote,
    QuasiQuote,
    Unquote,
    UnquoteSplicing,
    /// `#(`
    VectorPrefix,
    /// `#vu8(`
    ByteVectorPrefix,
    Boolean(bool),
    Character(char),
    Number(Number),
    String(String),
    /// `#;` datum comment marker (the following datum is to be skipped).
    DatumComment,
    Eof,
    /// A bare symbol/identifier.
    Symbol(String),
}

/// A token together with its source span.
#[derive(Debug, Clone, PartialEq)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

/// A lexical error.
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lex error at line {}, column {}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

impl std::error::Error for LexError {}

/// Tokenize the input source into a flat list of spanned tokens.
///
/// Comments (other than datum comments, which are emitted as a token
/// so the parser can consume the following datum) and whitespace are
/// consumed but not emitted. A single trailing [`Token::Eof`] is
/// appended.
pub fn tokenize(src: &str) -> Result<Vec<SpannedToken>, LexError> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut line: u32 = 1;
    let mut col: u32 = 1;
    let mut out: Vec<SpannedToken> = Vec::new();

    let bump = |i: &mut usize, col: &mut u32| {
        *i += 1;
        *col += 1;
    };
    let bump_newline = |i: &mut usize, line: &mut u32, col: &mut u32| {
        *i += 1;
        *line += 1;
        *col = 1;
    };

    while i < bytes.len() {
        let b = bytes[i];
        match b {
            // whitespace
            b' ' | b'\t' | b'\r' => {
                bump(&mut i, &mut col);
            }
            b'\n' => bump_newline(&mut i, &mut line, &mut col),
            // line comment
            b';' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            // hash-introduced tokens
            b'#' => {
                let span = Span { line, column: col, offset: i };
                if bytes.get(i + 1) == Some(&b'|') {
                    // nested block comment #| ... |#
                    i += 2;
                    col += 2;
                    let mut depth = 1usize;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == b'#' && bytes.get(i + 1) == Some(&b'|') {
                            depth += 1;
                            i += 2;
                            col += 2;
                        } else if bytes[i] == b'|' && bytes.get(i + 1) == Some(&b'#') {
                            depth -= 1;
                            i += 2;
                            col += 2;
                        } else if bytes[i] == b'\n' {
                            bump_newline(&mut i, &mut line, &mut col);
                        } else {
                            bump(&mut i, &mut col);
                        }
                    }
                    if depth != 0 {
                        return Err(LexError {
                            message: "unterminated block comment".to_string(),
                            span,
                        });
                    }
                    continue;
                }
                if bytes.get(i + 1) == Some(&b';') {
                    out.push(SpannedToken { token: Token::DatumComment, span });
                    i += 2;
                    col += 2;
                    continue;
                }
                if bytes.get(i + 1) == Some(&b'(') {
                    out.push(SpannedToken { token: Token::VectorPrefix, span });
                    i += 2;
                    col += 2;
                    continue;
                }
                if bytes.get(i + 1) == Some(&b'\\') {
                    // character literal
                    i += 2;
                    col += 2;
                    let (c, consumed) = read_character(bytes, i)?;
                    if c == '\n' {
                        line += 1;
                        col = 1;
                    } else {
                        col += consumed as u32;
                    }
                    i += consumed;
                    out.push(SpannedToken { token: Token::Character(c), span });
                    continue;
                }
                // #vu8(  (and #vU8(, #Vu8(, #VU8( per common impls)
                if let Some(rest) = read_bytevector_prefix(bytes, i) {
                    out.push(SpannedToken { token: Token::ByteVectorPrefix, span });
                    let advance = rest - i;
                    i = rest;
                    col += advance as u32;
                    continue;
                }
                // boolean #t #f #true #false
                if let Some((val, consumed)) = read_boolean(bytes, i)? {
                    out.push(SpannedToken { token: Token::Boolean(val), span });
                    i += consumed;
                    col += consumed as u32;
                    continue;
                }
                // Otherwise: a symbol beginning with # (e.g. #\x handled above;
                // bare # is unusual but let it fall through to symbol).
                let (sym, consumed) = read_symbol(bytes, i)?;
                if sym.is_empty() {
                    return Err(LexError {
                        message: format!("unexpected character `{}`", b as char),
                        span,
                    });
                }
                col += consumed as u32;
                i += consumed;
                out.push(SpannedToken { token: Token::Symbol(sym), span });
            }
            b'(' => {
                out.push(SpannedToken {
                    token: Token::LeftParen,
                    span: Span { line, column: col, offset: i },
                });
                bump(&mut i, &mut col);
            }
            b')' => {
                out.push(SpannedToken {
                    token: Token::RightParen,
                    span: Span { line, column: col, offset: i },
                });
                bump(&mut i, &mut col);
            }
            b'\'' => {
                out.push(SpannedToken {
                    token: Token::Quote,
                    span: Span { line, column: col, offset: i },
                });
                bump(&mut i, &mut col);
            }
            b'`' => {
                out.push(SpannedToken {
                    token: Token::QuasiQuote,
                    span: Span { line, column: col, offset: i },
                });
                bump(&mut i, &mut col);
            }
            b',' => {
                if bytes.get(i + 1) == Some(&b'@') {
                    out.push(SpannedToken {
                        token: Token::UnquoteSplicing,
                        span: Span { line, column: col, offset: i },
                    });
                    i += 2;
                    col += 2;
                } else {
                    out.push(SpannedToken {
                        token: Token::Unquote,
                        span: Span { line, column: col, offset: i },
                    });
                    bump(&mut i, &mut col);
                }
            }
            b'"' => {
                let span = Span { line, column: col, offset: i };
                i += 1;
                col += 1;
                let (s, new_i, lines_advanced) = read_string(bytes, &mut i, &mut line, &mut col)?;
                let _ = (new_i, lines_advanced);
                out.push(SpannedToken { token: Token::String(s), span });
                let _ = span;
            }
            // numbers: digits and sign+digit and special floats like +inf.0
            b'0'..=b'9' | b'+' | b'-' | b'.' => {
                if b == b'.' && is_symbol_terminator(bytes.get(i + 1).copied()) {
                    // lone dot = dotted-pair marker
                    out.push(SpannedToken {
                        token: Token::Dot,
                        span: Span { line, column: col, offset: i },
                    });
                    bump(&mut i, &mut col);
                    continue;
                }
                let span = Span { line, column: col, offset: i };
                let start = i;
                // Try to lex a number first; if that fails, treat as symbol.
                match read_number(bytes, start) {
                    Ok((num, consumed)) => {
                        col += consumed as u32;
                        i = start + consumed;
                        out.push(SpannedToken { token: Token::Number(num), span });
                    }
                    Err(_) => {
                        let (sym, consumed) = read_symbol(bytes, start)?;
                        if sym.is_empty() {
                            return Err(LexError {
                                message: format!("unexpected character `{}`", b as char),
                                span,
                            });
                        }
                        col += consumed as u32;
                        i = start + consumed;
                        out.push(SpannedToken { token: Token::Symbol(sym), span });
                    }
                }
            }
            _ => {
                let span = Span { line, column: col, offset: i };
                let start = i;
                let (sym, consumed) = read_symbol(bytes, start)?;
                if sym.is_empty() {
                    return Err(LexError {
                        message: format!("unexpected character `{}`", b as char),
                        span,
                    });
                }
                col += consumed as u32;
                i = start + consumed;
                out.push(SpannedToken { token: Token::Symbol(sym), span });
            }
        }
    }

    out.push(SpannedToken {
        token: Token::Eof,
        span: Span { line, column: col, offset: i },
    });
    Ok(out)
}

fn is_symbol_terminator(next: Option<u8>) -> bool {
    match next {
        None => true,
        Some(b) => matches!(
            b,
            b' ' | b'\t' | b'\r' | b'\n' | b'(' | b')' | b'"' | b';' | b'\'' | b'`' | b','
        ),
    }
}

/// Read a symbol starting at index `start`. Returns the symbol text and
/// the number of bytes consumed.
fn read_symbol(bytes: &[u8], start: usize) -> Result<(String, usize), LexError> {
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        if is_symbol_terminator(Some(b)) {
            break;
        }
        // handle inline hex escapes \x...;
        if b == b'\\' && bytes.get(i + 1) == Some(&b'x') {
            // consume up to the semicolon
            let mut j = i + 2;
            while j < bytes.len() && bytes[j] != b';' {
                j += 1;
            }
            if j >= bytes.len() {
                return Err(LexError {
                    message: "unterminated inline hex escape".to_string(),
                    span: Span { line: 0, column: 0, offset: i },
                });
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    let text = String::from_utf8_lossy(&bytes[start..i]).to_string();
    Ok((text, i - start))
}

/// Try to read a number at `start`. Returns Ok with the number and byte
/// length consumed, or Err if it is not a number.
fn read_number(bytes: &[u8], start: usize) -> Result<(Number, usize), ()> {
    let mut i = start;
    let mut consumed_digits = false;
    let mut has_dot = false;
    let mut has_e = false;
    let mut has_hash = false;
    let len = bytes.len();

    // optional sign
    if matches!(bytes.get(i), Some(b'+') | Some(b'-')) {
        i += 1;
    }
    while i < len {
        match bytes[i] {
            b'0'..=b'9' => {
                consumed_digits = true;
                i += 1;
            }
            b'.' if !has_dot && !has_e => {
                has_dot = true;
                i += 1;
            }
            b'e' | b'E' if consumed_digits && !has_e => {
                has_e = true;
                i += 1;
                if matches!(bytes.get(i), Some(b'+') | Some(b'-')) {
                    i += 1;
                }
            }
            // R6RS allows # in numbers as digit placeholder for inexactness
            b'#' if consumed_digits => {
                has_hash = true;
                i += 1;
            }
            _ => break,
        }
    }
    if !consumed_digits {
        return Err(());
    }
    let slice = String::from_utf8_lossy(&bytes[start..i]);
    let consumed = i - start;

    // Prefer exact integers when no fractional/exponent/inexactness marker is
    // present; only fall back to inexact for floats and special values.
    if !has_dot && !has_e && !has_hash {
        if let Ok(n) = slice.parse::<i64>() {
            return Ok((Number::Fixnum(n), consumed));
        }
    }
    if let Ok(f) = slice.parse::<f64>() {
        return Ok((Number::Flonum(f), consumed));
    }
    Err(())
}

/// Read a boolean literal starting at `start` (which begins with `#`).
/// Returns `(value, bytes_consumed)` for #t/#f/#true/#false.
fn read_boolean(bytes: &[u8], start: usize) -> Result<Option<(bool, usize)>, LexError> {
    let rest = &bytes[start + 1..];
    for (text, val) in [("true", true), ("false", false), ("t", true), ("f", false)] {
        let t = text.as_bytes();
        if rest.starts_with(t) {
            let after = rest.get(t.len()).copied();
            if is_symbol_terminator(after) {
                return Ok(Some((val, 1 + t.len())));
            }
        }
    }
    Ok(None)
}

/// Try to read a `#vu8(` (case-insensitive) bytevector prefix at `start`.
/// Returns the index just past `(` if matched, else `None`.
fn read_bytevector_prefix(bytes: &[u8], start: usize) -> Option<usize> {
    let b = bytes.get(start)?;
    if *b != b'#' {
        return None;
    }
    let lower = |x: u8| x.to_ascii_lowercase();
    if lower(bytes.get(start + 1).copied().unwrap_or(0)) != b'v'
        || lower(bytes.get(start + 2).copied().unwrap_or(0)) != b'u'
        || lower(bytes.get(start + 3).copied().unwrap_or(0)) != b'8'
        || bytes.get(start + 4).copied() != Some(b'(')
    {
        return None;
    }
    Some(start + 5)
}

/// Read a character literal body after `#\`. Returns the character and the
/// number of bytes consumed.
fn read_character(bytes: &[u8], start: usize) -> Result<(char, usize), LexError> {
    // Single ASCII char by default.
    if start >= bytes.len() {
        return Err(LexError {
            message: "unexpected end of input in character literal".to_string(),
            span: Span { line: 0, column: 0, offset: start },
        });
    }
    // Try a named character first if alphabetic and followed by more letters.
    if bytes[start].is_ascii_alphabetic() {
        let mut j = start;
        while j < bytes.len()
            && bytes[j].is_ascii_alphabetic()
        {
            j += 1;
        }
        if j > start + 1 {
            let name = String::from_utf8_lossy(&bytes[start..j]);
            if let Some(c) = named_character(&name) {
                return Ok((c, j - start));
            }
        }
    }
    // Otherwise exactly one byte -> char.
    let c = bytes[start] as char;
    Ok((c, 1))
}

fn named_character(name: &str) -> Option<char> {
    Some(match name {
        "nul" => '\u{0000}',
        "alarm" => '\u{0007}',
        "backspace" => '\u{0008}',
        "tab" => '\t',
        "linefeed" | "newline" | "lf" => '\n',
        "vtab" => '\u{000B}',
        "page" => '\u{000C}',
        "return" | "cr" => '\r',
        "esc" => '\u{001B}',
        "space" => ' ',
        "delete" => '\u{007F}',
        _ => return None,
    })
}

/// Read a string literal starting at `"` (already consumed by caller
/// logic). Advances `i`/`line`/`col` through the closing quote.
fn read_string(
    bytes: &[u8],
    i: &mut usize,
    _line: &mut u32,
    _col: &mut u32,
) -> Result<(String, usize, u32), LexError> {
    let mut out = String::new();
    let start_offset = *i;
    while *i < bytes.len() {
        let b = bytes[*i];
        if b == b'"' {
            *i += 1;
            return Ok((out, *i - start_offset, 0));
        }
        if b == b'\\' {
            *i += 1;
            if *i >= bytes.len() {
                return Err(LexError {
                    message: "unterminated string escape".to_string(),
                    span: Span { line: 0, column: 0, offset: start_offset },
                });
            }
            let esc = bytes[*i];
            *i += 1;
            match esc {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'n' => out.push('\n'),
                b't' => out.push('\t'),
                b'r' => out.push('\r'),
                b'a' => out.push('\u{0007}'),
                b'b' => out.push('\u{0008}'),
                b'f' => out.push('\u{000C}'),
                b'v' => out.push('\u{000B}'),
                b'x' => {
                    // \x...; hex escape
                    let mut value: u32 = 0;
                    let mut saw = false;
                    while *i < bytes.len() && bytes[*i] != b';' {
                        let h = bytes[*i];
                        let d = match h {
                            b'0'..=b'9' => (h - b'0') as u32,
                            b'a'..=b'f' => (h - b'a' + 10) as u32,
                            b'A'..=b'F' => (h - b'A' + 10) as u32,
                            _ => {
                                return Err(LexError {
                                    message: "invalid hex escape in string".to_string(),
                                    span: Span { line: 0, column: 0, offset: *i },
                                })
                            }
                        };
                        value = value * 16 + d;
                        saw = true;
                        *i += 1;
                    }
                    if !saw || *i >= bytes.len() {
                        return Err(LexError {
                            message: "malformed hex escape in string".to_string(),
                            span: Span { line: 0, column: 0, offset: *i },
                        });
                    }
                    *i += 1; // consume ';'
                    if let Some(c) = char::from_u32(value) {
                        out.push(c);
                    } else {
                        return Err(LexError {
                            message: "invalid scalar value in string escape".to_string(),
                            span: Span { line: 0, column: 0, offset: *i },
                        });
                    }
                }
                _ => {
                    return Err(LexError {
                        message: format!("unknown string escape \\{}", esc as char),
                        span: Span { line: 0, column: 0, offset: *i },
                    })
                }
            }
        } else {
            out.push(b as char);
            *i += 1;
        }
    }
    Err(LexError {
        message: "unterminated string".to_string(),
        span: Span { line: 0, column: 0, offset: start_offset },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn just(src: &str) -> Vec<Token> {
        let mut tokens = tokenize(src).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(tokens.last().unwrap().token, Token::Eof));
        tokens.pop();
        tokens.into_iter().map(|st| st.token).collect()
    }

    #[test]
    fn lex_empty_and_eof() {
        let t = tokenize("").unwrap();
        assert_eq!(t.len(), 1);
        assert!(matches!(t[0].token, Token::Eof));
    }

    #[test]
    fn lex_whitespace_only() {
        assert_eq!(just("   \n\t  \n"), vec![]);
    }

    #[test]
    fn lex_line_comment() {
        assert_eq!(just("; a comment\n  ;another\n"), vec![]);
    }

    #[test]
    fn lex_block_comment_nest() {
        assert_eq!(just("#| outer |#"), vec![]);
        assert_eq!(just("#| a #| b |# c |#"), vec![]);
    }

    #[test]
    fn lex_block_comment_unterminated_errors() {
        assert!(tokenize("#| open").is_err());
    }

    #[test]
    fn lex_parens_and_dot() {
        assert_eq!(just("( . )"), vec![
            Token::LeftParen,
            Token::Dot,
            Token::RightParen,
        ]);
    }

    #[test]
    fn lex_quote_abbrevs() {
        assert_eq!(just("'`,"), vec![
            Token::Quote,
            Token::QuasiQuote,
            Token::Unquote,
        ]);
        assert_eq!(just(",@"), vec![Token::UnquoteSplicing]);
    }

    #[test]
    fn lex_booleans() {
        assert_eq!(just("#t #f #true #false"), vec![
            Token::Boolean(true),
            Token::Boolean(false),
            Token::Boolean(true),
            Token::Boolean(false),
        ]);
    }

    #[test]
    fn lex_characters() {
        assert_eq!(just("#\\a #\\space #\\newline #\\("), vec![
            Token::Character('a'),
            Token::Character(' '),
            Token::Character('\n'),
            Token::Character('('),
        ]);
    }

    #[test]
    fn lex_strings() {
        assert_eq!(just("\"abc\" \"a\\nb\""), vec![
            Token::String("abc".to_string()),
            Token::String("a\nb".to_string()),
        ]);
        assert_eq!(just("\"\\x41;\""), vec![Token::String("A".to_string())]);
        assert_eq!(just("\"a\\\"b\""), vec![Token::String("a\"b".to_string())]);
    }

    #[test]
    fn lex_string_unterminated_errors() {
        assert!(tokenize("\"abc").is_err());
        assert!(tokenize("\"a\\xZ;\"").is_err());
    }

    #[test]
    fn lex_fixnums() {
        assert_eq!(just("0 42 -7 +100"), vec![
            Token::Number(Number::Fixnum(0)),
            Token::Number(Number::Fixnum(42)),
            Token::Number(Number::Fixnum(-7)),
            Token::Number(Number::Fixnum(100)),
        ]);
    }

    #[test]
    fn lex_flonums() {
        assert_eq!(just("3.14 -0.5 1e10 2.5e-3"), vec![
            Token::Number(Number::Flonum(3.14)),
            Token::Number(Number::Flonum(-0.5)),
            Token::Number(Number::Flonum(1e10)),
            Token::Number(Number::Flonum(2.5e-3)),
        ]);
    }

    #[test]
    fn lex_symbols() {
        assert_eq!(just("+ - * < <= = => set! list->vector"), vec![
            Token::Symbol("+".to_string()),
            Token::Symbol("-".to_string()),
            Token::Symbol("*".to_string()),
            Token::Symbol("<".to_string()),
            Token::Symbol("<=".to_string()),
            Token::Symbol("=".to_string()),
            Token::Symbol("=>".to_string()),
            Token::Symbol("set!".to_string()),
            Token::Symbol("list->vector".to_string()),
        ]);
    }

    #[test]
    fn lex_vector_and_bytevector_prefix() {
        assert_eq!(just("#( #vu8("), vec![
            Token::VectorPrefix,
            Token::ByteVectorPrefix,
        ]);
        assert_eq!(just("#VU8("), vec![Token::ByteVectorPrefix]);
    }

    #[test]
    fn lex_datum_comment() {
        assert_eq!(just("#;x"), vec![Token::DatumComment, Token::Symbol("x".to_string())]);
    }

    #[test]
    fn lex_mixed_list() {
        assert_eq!(
            just("(define x (+ 1 2))"),
            vec![
                Token::LeftParen,
                Token::Symbol("define".to_string()),
                Token::Symbol("x".to_string()),
                Token::LeftParen,
                Token::Symbol("+".to_string()),
                Token::Number(Number::Fixnum(1)),
                Token::Number(Number::Fixnum(2)),
                Token::RightParen,
                Token::RightParen,
            ]
        );
    }
}
