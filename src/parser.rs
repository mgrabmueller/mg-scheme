//! Parser for R6RS Scheme.
//!
//! Consumes the flat token stream produced by [`crate::lexer::tokenize`]
//! and builds the surface-syntax [`crate::ast::Expr`] tree. It first reads
//! raw data via the *datum* layer (lists, vectors, bytevectors, quote
//! abbreviations), then a second pass recognizes the special syntactic
//! forms (`lambda`, `if`, `let`, ...). Datums that are not recognized as a
//! special form become a variable reference or a procedure application.
//!
//! The parser is split this way so that `quote` and `case` clauses can
//! reuse the datum layer directly, while expressions are layered on top.

use crate::ast::{
    Binding, Body, CaseClause, CondClause, Datum, Definition, Expr, Formals, Ident, MvBinding,
    Number, TopLevelForm,
};
use crate::lexer::{tokenize, LexError, Span, SpannedToken, Token};

/// A parsing error.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "parse error at line {}, column {}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

impl std::error::Error for ParseError {}

impl From<LexError> for ParseError {
    fn from(e: LexError) -> Self {
        ParseError {
            message: e.message,
            span: e.span,
        }
    }
}

/// Parse a full program (a sequence of top-level forms) from source.
pub fn parse_program(src: &str) -> Result<crate::ast::Program, ParseError> {
    let tokens = tokenize(src)?;
    let mut p = Parser::new(tokens);
    let mut forms = Vec::new();
    while !p.at_end() {
        forms.push(p.parse_top_level_form()?);
    }
    Ok(crate::ast::Program { forms })
}

/// Parse a single expression from source (useful for tests / REPL).
pub fn parse_expr(src: &str) -> Result<Expr, ParseError> {
    let tokens = tokenize(src)?;
    let mut p = Parser::new(tokens);
    let e = p.parse_expr()?;
    if !p.at_end() {
        return Err(p.error("unexpected trailing input"));
    }
    Ok(e)
}

struct Parser {
    toks: Vec<SpannedToken>,
    pos: usize,
}

impl Parser {
    fn new(toks: Vec<SpannedToken>) -> Self {
        Parser { toks, pos: 0 }
    }

    fn peek(&self) -> &Token {
        &self.toks[self.pos].token
    }

    fn peek_span(&self) -> Span {
        self.toks[self.pos].span
    }

    fn at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let t = self.toks[self.pos].token.clone();
        if !matches!(t, Token::Eof) {
            self.pos += 1;
        }
        t
    }

    fn error(&self, msg: &str) -> ParseError {
        ParseError {
            message: msg.to_string(),
            span: self.peek_span(),
        }
    }

    fn expect(&mut self, want: Token, msg: &str) -> Result<Token, ParseError> {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(&want) {
            Ok(self.advance())
        } else {
            Err(self.error(msg))
        }
    }

    // ---- Top level ----

    fn parse_top_level_form(&mut self) -> Result<TopLevelForm, ParseError> {
        // handle datum comments at the top level: skip the next datum
        while matches!(self.peek(), Token::DatumComment) {
            self.advance();
            let _ = self.parse_datum()?;
        }
        // A top-level definition/begin is always parenthesized: (define ...),
        // (begin ...). Peek one token past the opening `(` to dispatch.
        if matches!(self.peek(), Token::LeftParen) {
            if let Some(Token::Symbol(inner)) = self.toks.get(self.pos + 1).map(|s| &s.token) {
                match inner.as_str() {
                    "define" | "define-syntax" => {
                        self.advance(); // (
                        let def = self.parse_definition()?;
                        self.expect(Token::RightParen, "expected ) to close definition")?;
                        return Ok(TopLevelForm::Definition(def));
                    }
                    "begin" => {
                        self.advance(); // (
                        self.advance(); // begin
                        let mut forms = Vec::new();
                        while !matches!(self.peek(), Token::RightParen | Token::Eof) {
                            forms.push(self.parse_top_level_form()?);
                        }
                        self.expect(Token::RightParen, "expected ) to close begin")?;
                        return Ok(TopLevelForm::Begin(forms));
                    }
                    _ => {}
                }
            }
        }
        let e = self.parse_expr()?;
        Ok(TopLevelForm::Expr(e))
    }

    // ---- Definitions ----

    fn parse_definition(&mut self) -> Result<Definition, ParseError> {
        let head = match self.peek().clone() {
            Token::Symbol(s) => s,
            _ => return Err(self.error("expected definition keyword")),
        };
        self.advance(); // consume define/define-syntax
        match head.as_str() {
            "define" => self.parse_define_rest(),
            "define-syntax" => {
                let kw = self.parse_ident()?;
                let init = self.parse_expr()?;
                Ok(Definition::Syntax {
                    keyword: kw,
                    init: Box::new(init),
                })
            }
            _ => Err(self.error("expected define or define-syntax")),
        }
    }

    fn parse_define_rest(&mut self) -> Result<Definition, ParseError> {
        // (define <var> <expr>?)            -> Variable
        // (define (<var> . <formals>) <body>) -> Function
        if matches!(self.peek(), Token::LeftParen) {
            self.advance();
            let name = self.parse_ident()?;
            let formals = self.parse_formals_rest()?;
            self.expect(Token::RightParen, "expected ) to close (define (name ...)")?;
            let body = self.parse_body()?;
            Ok(Definition::Function { name, formals, body })
        } else {
            let name = self.parse_ident()?;
            if matches!(
                self.peek(),
                Token::RightParen | Token::Eof | Token::DatumComment
            ) || self.at_end_of_form()
            {
                Ok(Definition::Variable { name, init: None })
            } else {
                let init = self.parse_expr()?;
                Ok(Definition::Variable {
                    name,
                    init: Some(Box::new(init)),
                })
            }
        }
    }

    fn at_end_of_form(&self) -> bool {
        matches!(self.peek(), Token::RightParen)
    }

    /// After the function name in `(define (name . <formals>) ...)`, parse
    /// the remaining fixed/rest formals. The closing `)` is handled by the
    /// caller.
    fn parse_formals_rest(&mut self) -> Result<Formals, ParseError> {
        let mut fixed = Vec::new();
        let mut rest = None;
        while !matches!(self.peek(), Token::RightParen | Token::Eof) {
            if matches!(self.peek(), Token::Dot) {
                self.advance();
                rest = Some(self.parse_ident()?);
                break;
            }
            fixed.push(self.parse_ident()?);
        }
        Ok(Formals { fixed, rest })
    }

    // ---- Body ----

    /// Parse a `<body>`: zero or more internal definitions followed by one
    /// or more expressions. Assumes the caller will consume the closing `)`.
    fn parse_body(&mut self) -> Result<Body, ParseError> {
        let mut definitions = Vec::new();
        // leading definitions
        loop {
            // skip datum comments interspersed
            let is_def = matches!(self.peek(), Token::Symbol(s) if s == "define" || s == "define-syntax");
            if !is_def {
                break;
            }
            definitions.push(self.parse_definition()?);
        }
        let mut expressions = Vec::new();
        while !matches!(self.peek(), Token::RightParen | Token::Eof) {
            while matches!(self.peek(), Token::DatumComment) {
                self.advance();
                let _ = self.parse_datum()?;
            }
            if matches!(self.peek(), Token::RightParen | Token::Eof) {
                break;
            }
            expressions.push(self.parse_expr()?);
        }
        if expressions.is_empty() {
            return Err(self.error("body requires at least one expression"));
        }
        Ok(Body {
            definitions,
            expressions,
        })
    }

    // ---- Expressions ----

    fn parse_ident(&mut self) -> Result<Ident, ParseError> {
        match self.peek().clone() {
            Token::Symbol(s) => {
                self.advance();
                Ok(Ident(s))
            }
            other => {
                let _ = other;
                Err(self.error("expected identifier"))
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        // skip datum comments before an expression
        while matches!(self.peek(), Token::DatumComment) {
            self.advance();
            let _ = self.parse_datum()?;
        }
        match self.peek().clone() {
            Token::Quote => {
                self.advance();
                let d = self.parse_datum()?;
                Ok(Expr::Quote(d))
            }
            Token::QuasiQuote => {
                // not in the core AST; represent as application of a `quasiquote`
                // procedure for now, since quasiquote is handled by macros.
                self.advance();
                let d = self.parse_datum()?;
                Ok(Expr::Application {
                    operator: Box::new(Expr::Var(Ident("quasiquote".to_string()))),
                    operands: vec![datum_to_expr(&d)?],
                })
            }
            Token::Unquote => {
                self.advance();
                let d = self.parse_datum()?;
                Ok(Expr::Application {
                    operator: Box::new(Expr::Var(Ident("unquote".to_string()))),
                    operands: vec![datum_to_expr(&d)?],
                })
            }
            Token::UnquoteSplicing => {
                self.advance();
                let d = self.parse_datum()?;
                Ok(Expr::Application {
                    operator: Box::new(Expr::Var(Ident("unquote-splicing".to_string()))),
                    operands: vec![datum_to_expr(&d)?],
                })
            }
            Token::LeftParen => self.parse_list_expr(),
            Token::VectorPrefix => {
                let d = self.parse_datum()?;
                datum_to_expr(&d)
            }
            Token::ByteVectorPrefix => {
                let d = self.parse_datum()?;
                datum_to_expr(&d)
            }
            Token::Boolean(b) => {
                self.advance();
                Ok(Expr::Literal(Datum::Boolean(b)))
            }
            Token::Character(c) => {
                self.advance();
                Ok(Expr::Literal(Datum::Character(c)))
            }
            Token::Number(n) => {
                self.advance();
                Ok(Expr::Literal(Datum::Number(n)))
            }
            Token::String(s) => {
                self.advance();
                Ok(Expr::Literal(Datum::String(s)))
            }
            Token::Symbol(s) => {
                self.advance();
                Ok(Expr::Var(Ident(s)))
            }
            Token::Eof => Err(self.error("unexpected end of input")),
            Token::RightParen => Err(self.error("unexpected )")),
            Token::Dot => Err(self.error("unexpected .")),
            Token::DatumComment => unreachable!(),
        }
    }

    /// Parse a parenthesized form: either a special form or a procedure
    /// application. The opening `(` has been seen by the caller? No: this
    /// is called with `(` as the current token.
    fn parse_list_expr(&mut self) -> Result<Expr, ParseError> {
        self.expect(Token::LeftParen, "expected (")?;
        if matches!(self.peek(), Token::RightParen) {
            self.advance();
            // () is an empty application of nothing, which is a syntax error
            // in R6RS; represent as a literal empty list datum via quote.
            return Ok(Expr::Quote(Datum::Null));
        }
        let head = match self.peek().clone() {
            Token::Symbol(s) => Some(s),
            _ => None,
        };
        if let Some(kw) = head {
            match kw.as_str() {
                "lambda" => {
                    self.advance();
                    return self.parse_lambda();
                }
                "if" => {
                    self.advance();
                    return self.parse_if();
                }
                "set!" => {
                    self.advance();
                    return self.parse_set();
                }
                "cond" => {
                    self.advance();
                    return self.parse_cond();
                }
                "case" => {
                    self.advance();
                    return self.parse_case();
                }
                "when" => {
                    self.advance();
                    return self.parse_when_unless(true);
                }
                "unless" => {
                    self.advance();
                    return self.parse_when_unless(false);
                }
                "and" => {
                    self.advance();
                    return self.parse_and_or(AndOr::And);
                }
                "or" => {
                    self.advance();
                    return self.parse_and_or(AndOr::Or);
                }
                "let" => {
                    self.advance();
                    return self.parse_let();
                }
                "let*" => {
                    self.advance();
                    return self.parse_let_star();
                }
                "letrec" => {
                    self.advance();
                    return self.parse_letrec();
                }
                "letrec*" => {
                    self.advance();
                    return self.parse_letrec_star();
                }
                "let-values" => {
                    self.advance();
                    return self.parse_let_values(false);
                }
                "let*-values" => {
                    self.advance();
                    return self.parse_let_values(true);
                }
                "begin" => {
                    self.advance();
                    let mut exprs = Vec::new();
                    while !matches!(self.peek(), Token::RightParen | Token::Eof) {
                        exprs.push(self.parse_expr()?);
                    }
                    self.expect(Token::RightParen, "expected ) to close begin")?;
                    return Ok(Expr::Begin(exprs));
                }
                _ => {}
            }
        }
        // otherwise: procedure application
        let operator = self.parse_expr()?;
        let mut operands = Vec::new();
        while !matches!(self.peek(), Token::RightParen | Token::Eof) {
            operands.push(self.parse_expr()?);
        }
        self.expect(Token::RightParen, "expected ) to close application")?;
        Ok(Expr::Application {
            operator: Box::new(operator),
            operands,
        })
    }

    fn parse_lambda(&mut self) -> Result<Expr, ParseError> {
        let formals = self.parse_formals()?;
        let body = self.parse_body()?;
        self.expect(Token::RightParen, "expected ) to close lambda")?;
        Ok(Expr::Lambda { formals, body })
    }

    /// Parse a formals spec: either a single identifier (fully variadic) or
    /// a parenthesized list possibly with a dotted rest.
    fn parse_formals(&mut self) -> Result<Formals, ParseError> {
        match self.peek().clone() {
            Token::Symbol(s) => {
                self.advance();
                Ok(Formals {
                    fixed: vec![],
                    rest: Some(Ident(s)),
                })
            }
            Token::LeftParen => {
                self.advance();
                let formals = self.parse_formals_rest()?;
                self.expect(Token::RightParen, "expected ) to close formals")?;
                Ok(formals)
            }
            _ => Err(self.error("expected formals")),
        }
    }

    fn parse_if(&mut self) -> Result<Expr, ParseError> {
        let test = self.parse_expr()?;
        let consequent = self.parse_expr()?;
        let alternate = if matches!(self.peek(), Token::RightParen) {
            None
        } else {
            Some(Box::new(self.parse_expr()?))
        };
        self.expect(Token::RightParen, "expected ) to close if")?;
        Ok(Expr::If {
            test: Box::new(test),
            consequent: Box::new(consequent),
            alternate,
        })
    }

    fn parse_set(&mut self) -> Result<Expr, ParseError> {
        let var = self.parse_ident()?;
        let value = self.parse_expr()?;
        self.expect(Token::RightParen, "expected ) to close set!")?;
        Ok(Expr::Set {
            var,
            value: Box::new(value),
        })
    }

    fn parse_cond(&mut self) -> Result<Expr, ParseError> {
        let mut clauses = Vec::new();
        while !matches!(self.peek(), Token::RightParen | Token::Eof) {
            self.expect(Token::LeftParen, "expected ( to open cond clause")?;
            if let Token::Symbol(s) = self.peek().clone() {
                if s == "else" {
                    self.advance();
                    let mut body = Vec::new();
                    while !matches!(self.peek(), Token::RightParen | Token::Eof) {
                        body.push(self.parse_expr()?);
                    }
                    self.expect(Token::RightParen, "expected ) to close else clause")?;
                    clauses.push(CondClause::Else { body });
                    continue;
                }
            }
            let test = self.parse_expr()?;
            if let Token::Symbol(s) = self.peek().clone() {
                if s == "=>" {
                    self.advance();
                    let recipient = self.parse_expr()?;
                    self.expect(Token::RightParen, "expected ) to close => clause")?;
                    clauses.push(CondClause::Arrow {
                        test: Box::new(test),
                        recipient: Box::new(recipient),
                    });
                    continue;
                }
            }
            let mut body = Vec::new();
            while !matches!(self.peek(), Token::RightParen | Token::Eof) {
                body.push(self.parse_expr()?);
            }
            self.expect(Token::RightParen, "expected ) to close cond clause")?;
            clauses.push(CondClause::Clause {
                test: Box::new(test),
                body,
            });
        }
        self.expect(Token::RightParen, "expected ) to close cond")?;
        Ok(Expr::Cond(clauses))
    }

    fn parse_case(&mut self) -> Result<Expr, ParseError> {
        let key = self.parse_expr()?;
        let mut clauses = Vec::new();
        while !matches!(self.peek(), Token::RightParen | Token::Eof) {
            self.expect(Token::LeftParen, "expected ( to open case clause")?;
            if let Token::Symbol(s) = self.peek().clone() {
                if s == "else" {
                    self.advance();
                    let mut body = Vec::new();
                    while !matches!(self.peek(), Token::RightParen | Token::Eof) {
                        body.push(self.parse_expr()?);
                    }
                    self.expect(Token::RightParen, "expected ) to close else clause")?;
                    clauses.push(CaseClause::Else { body });
                    continue;
                }
            }
            // ((d ...) body...)
            self.expect(Token::LeftParen, "expected ( to open datum list")?;
            let mut data = Vec::new();
            while !matches!(self.peek(), Token::RightParen | Token::Eof) {
                data.push(self.parse_datum()?);
            }
            self.expect(Token::RightParen, "expected ) to close datum list")?;
            let mut body = Vec::new();
            while !matches!(self.peek(), Token::RightParen | Token::Eof) {
                body.push(self.parse_expr()?);
            }
            self.expect(Token::RightParen, "expected ) to close case clause")?;
            clauses.push(CaseClause::Matches { data, body });
        }
        self.expect(Token::RightParen, "expected ) to close case")?;
        Ok(Expr::Case {
            key: Box::new(key),
            clauses,
        })
    }

    fn parse_when_unless(&mut self, is_when: bool) -> Result<Expr, ParseError> {
        let test = self.parse_expr()?;
        let mut body = Vec::new();
        while !matches!(self.peek(), Token::RightParen | Token::Eof) {
            body.push(self.parse_expr()?);
        }
        self.expect(Token::RightParen, "expected ) to close when/unless")?;
        let body = body;
        if is_when {
            Ok(Expr::When {
                test: Box::new(test),
                body,
            })
        } else {
            Ok(Expr::Unless {
                test: Box::new(test),
                body,
            })
        }
    }

    fn parse_and_or(&mut self, which: AndOr) -> Result<Expr, ParseError> {
        let mut exprs = Vec::new();
        while !matches!(self.peek(), Token::RightParen | Token::Eof) {
            exprs.push(self.parse_expr()?);
        }
        self.expect(Token::RightParen, "expected ) to close and/or")?;
        Ok(match which {
            AndOr::And => Expr::And(exprs),
            AndOr::Or => Expr::Or(exprs),
        })
    }

    fn parse_let(&mut self) -> Result<Expr, ParseError> {
        // named let: (let name ((var init)...) body...)
        let name = if let Token::Symbol(s) = self.peek().clone() {
            if !s.is_empty() && !is_pure_number(&s) {
                // peek further: a named let has a symbol then `(`
                self.advance();
                Some(Ident(s))
            } else {
                None
            }
        } else {
            None
        };
        let bindings = self.parse_bindings()?;
        let body = self.parse_body()?;
        self.expect(Token::RightParen, "expected ) to close let")?;
        Ok(Expr::Let {
            name,
            bindings,
            body,
        })
    }

    fn parse_let_star(&mut self) -> Result<Expr, ParseError> {
        let bindings = self.parse_bindings()?;
        let body = self.parse_body()?;
        self.expect(Token::RightParen, "expected ) to close let*")?;
        Ok(Expr::LetStar { bindings, body })
    }

    fn parse_letrec(&mut self) -> Result<Expr, ParseError> {
        let bindings = self.parse_bindings()?;
        let body = self.parse_body()?;
        self.expect(Token::RightParen, "expected ) to close letrec")?;
        Ok(Expr::LetRec { bindings, body })
    }

    fn parse_letrec_star(&mut self) -> Result<Expr, ParseError> {
        let bindings = self.parse_bindings()?;
        let body = self.parse_body()?;
        self.expect(Token::RightParen, "expected ) to close letrec*")?;
        Ok(Expr::LetRecStar { bindings, body })
    }

    fn parse_let_values(&mut self, star: bool) -> Result<Expr, ParseError> {
        let bindings = self.parse_mv_bindings()?;
        let body = self.parse_body()?;
        let closer = if star { "let*-values" } else { "let-values" };
        self.expect(Token::RightParen, "expected ) to close")?;
        let _ = closer;
        Ok(if star {
            Expr::LetStarValues { bindings, body }
        } else {
            Expr::LetValues { bindings, body }
        })
    }

    fn parse_bindings(&mut self) -> Result<Vec<Binding>, ParseError> {
        self.expect(Token::LeftParen, "expected ( to open bindings")?;
        let mut bindings = Vec::new();
        while !matches!(self.peek(), Token::RightParen | Token::Eof) {
            self.expect(Token::LeftParen, "expected ( to open binding")?;
            let name = self.parse_ident()?;
            let init = self.parse_expr()?;
            self.expect(Token::RightParen, "expected ) to close binding")?;
            bindings.push(Binding {
                name,
                init: Box::new(init),
            });
        }
        self.expect(Token::RightParen, "expected ) to close bindings")?;
        Ok(bindings)
    }

    fn parse_mv_bindings(&mut self) -> Result<Vec<MvBinding>, ParseError> {
        self.expect(Token::LeftParen, "expected ( to open mv bindings")?;
        let mut bindings = Vec::new();
        while !matches!(self.peek(), Token::RightParen | Token::Eof) {
            self.expect(Token::LeftParen, "expected ( to open mv binding")?;
            let formals = self.parse_formals()?;
            let init = self.parse_expr()?;
            self.expect(Token::RightParen, "expected ) to close mv binding")?;
            bindings.push(MvBinding {
                formals,
                init: Box::new(init),
            });
        }
        self.expect(Token::RightParen, "expected ) to close mv bindings")?;
        Ok(bindings)
    }

    // ---- Datum layer ----

    fn parse_datum(&mut self) -> Result<Datum, ParseError> {
        while matches!(self.peek(), Token::DatumComment) {
            self.advance();
            let _ = self.parse_datum()?;
        }
        match self.peek().clone() {
            Token::Boolean(b) => {
                self.advance();
                Ok(Datum::Boolean(b))
            }
            Token::Character(c) => {
                self.advance();
                Ok(Datum::Character(c))
            }
            Token::Number(n) => {
                self.advance();
                Ok(Datum::Number(n))
            }
            Token::String(s) => {
                self.advance();
                Ok(Datum::String(s))
            }
            Token::Symbol(s) => {
                self.advance();
                Ok(Datum::Symbol(s))
            }
            Token::Quote => {
                self.advance();
                let d = self.parse_datum()?;
                Ok(Datum::Pair(
                    Box::new(Datum::Symbol("quote".to_string())),
                    Box::new(Datum::Pair(Box::new(d), Box::new(Datum::Null))),
                ))
            }
            Token::QuasiQuote => {
                self.advance();
                let d = self.parse_datum()?;
                Ok(Datum::Pair(
                    Box::new(Datum::Symbol("quasiquote".to_string())),
                    Box::new(Datum::Pair(Box::new(d), Box::new(Datum::Null))),
                ))
            }
            Token::Unquote => {
                self.advance();
                let d = self.parse_datum()?;
                Ok(Datum::Pair(
                    Box::new(Datum::Symbol("unquote".to_string())),
                    Box::new(Datum::Pair(Box::new(d), Box::new(Datum::Null))),
                ))
            }
            Token::UnquoteSplicing => {
                self.advance();
                let d = self.parse_datum()?;
                Ok(Datum::Pair(
                    Box::new(Datum::Symbol("unquote-splicing".to_string())),
                    Box::new(Datum::Pair(Box::new(d), Box::new(Datum::Null))),
                ))
            }
            Token::VectorPrefix => {
                self.advance();
                let mut items = Vec::new();
                while !matches!(self.peek(), Token::RightParen | Token::Eof) {
                    items.push(self.parse_datum()?);
                }
                self.expect(Token::RightParen, "expected ) to close vector")?;
                Ok(Datum::Vector(items))
            }
            Token::ByteVectorPrefix => {
                self.advance();
                let mut bytes = Vec::new();
                while !matches!(self.peek(), Token::RightParen | Token::Eof) {
                    match self.parse_datum()? {
                        Datum::Number(Number::Fixnum(n)) if (0..=255).contains(&n) => {
                            bytes.push(n as u8);
                        }
                        _ => return Err(self.error("bytevector must contain exact uint8")),
                    }
                }
                self.expect(Token::RightParen, "expected ) to close bytevector")?;
                Ok(Datum::ByteVector(bytes))
            }
            Token::LeftParen => self.parse_list_datum(),
            Token::RightParen => Err(self.error("unexpected )")),
            Token::Dot => Err(self.error("unexpected .")),
            Token::Eof => Err(self.error("unexpected end of input")),
            Token::DatumComment => unreachable!(),
        }
    }

    fn parse_list_datum(&mut self) -> Result<Datum, ParseError> {
        self.expect(Token::LeftParen, "expected (")?;
        if matches!(self.peek(), Token::RightParen) {
            self.advance();
            return Ok(Datum::Null);
        }
        let mut items = Vec::new();
        let mut tail = Datum::Null;
        loop {
            if matches!(self.peek(), Token::Dot) {
                self.advance();
                tail = self.parse_datum()?;
                break;
            }
            if matches!(self.peek(), Token::RightParen | Token::Eof) {
                break;
            }
            items.push(self.parse_datum()?);
        }
        self.expect(Token::RightParen, "expected ) to close list")?;
        // build right-associated pair chain
        let mut list = tail;
        for d in items.into_iter().rev() {
            list = Datum::Pair(Box::new(d), Box::new(list));
        }
        Ok(list)
    }
}

enum AndOr {
    And,
    Or,
}

fn is_pure_number(s: &str) -> bool {
    s.parse::<f64>().is_ok() || s.parse::<i64>().is_ok()
}

/// Convert a quoted datum into an expression. Used for `quote` and the
/// quasiquote family, where a datum must be reified as an `Expr::Literal`.
fn datum_to_expr(d: &Datum) -> Result<Expr, ParseError> {
    Ok(Expr::Literal(d.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Number, TopLevelForm};

    fn id(s: &str) -> Ident {
        Ident(s.to_string())
    }

    fn parse_ok(src: &str) -> Expr {
        parse_expr(src).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn parse_number_literal() {
        assert_eq!(parse_ok("42"), Expr::Literal(Datum::Number(Number::Fixnum(42))));
        assert_eq!(
            parse_ok("3.14"),
            Expr::Literal(Datum::Number(Number::Flonum(3.14)))
        );
    }

    #[test]
    fn parse_boolean_and_char_and_string() {
        assert_eq!(parse_ok("#t"), Expr::Literal(Datum::Boolean(true)));
        assert_eq!(parse_ok("#\\a"), Expr::Literal(Datum::Character('a')));
        assert_eq!(
            parse_ok("\"hi\""),
            Expr::Literal(Datum::String("hi".to_string()))
        );
    }

    #[test]
    fn parse_var() {
        assert_eq!(parse_ok("x"), Expr::Var(id("x")));
        assert_eq!(parse_ok("+"), Expr::Var(id("+")));
    }

    #[test]
    fn parse_quote() {
        assert_eq!(
            parse_ok("'a"),
            Expr::Quote(Datum::Symbol("a".to_string()))
        );
        assert_eq!(
            parse_ok("'(1 2 3)"),
            Expr::Quote(list(&[
                Datum::Number(Number::Fixnum(1)),
                Datum::Number(Number::Fixnum(2)),
                Datum::Number(Number::Fixnum(3)),
            ]))
        );
    }

    #[test]
    fn parse_application() {
        assert_eq!(
            parse_ok("(+ 1 2)"),
            Expr::Application {
                operator: Box::new(Expr::Var(id("+"))),
                operands: vec![
                    Expr::Literal(Datum::Number(Number::Fixnum(1))),
                    Expr::Literal(Datum::Number(Number::Fixnum(2))),
                ],
            }
        );
    }

    #[test]
    fn parse_lambda_fixed_and_rest() {
        let e = parse_ok("(lambda (x) x)");
        match e {
            Expr::Lambda { formals, body } => {
                assert_eq!(formals.fixed, vec![id("x")]);
                assert!(formals.rest.is_none());
                assert_eq!(body.expressions.len(), 1);
            }
            _ => panic!("expected lambda"),
        }
        let e2 = parse_ok("(lambda (a b . rest) rest)");
        match e2 {
            Expr::Lambda { formals, .. } => {
                assert_eq!(formals.fixed, vec![id("a"), id("b")]);
                assert_eq!(formals.rest, Some(id("rest")));
            }
            _ => panic!("expected lambda"),
        }
        let e3 = parse_ok("(lambda args args)");
        match e3 {
            Expr::Lambda { formals, .. } => {
                assert!(formals.fixed.is_empty());
                assert_eq!(formals.rest, Some(id("args")));
            }
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn parse_if_two_and_three_arm() {
        let two = parse_ok("(if t c)");
        match two {
            Expr::If { alternate: None, .. } => {}
            _ => panic!("expected if without alternate"),
        }
        let three = parse_ok("(if t c a)");
        match three {
            Expr::If {
                alternate: Some(_), ..
            } => {}
            _ => panic!("expected if with alternate"),
        }
    }

    #[test]
    fn parse_set_bang() {
        assert_eq!(
            parse_ok("(set! x 5)"),
            Expr::Set {
                var: id("x"),
                value: Box::new(Expr::Literal(Datum::Number(Number::Fixnum(5)))),
            }
        );
    }

    #[test]
    fn parse_cond_with_arrow_and_else() {
        let e = parse_ok("(cond (a => f) (else b))");
        match e {
            Expr::Cond(clauses) => {
                assert_eq!(clauses.len(), 2);
                assert!(matches!(clauses[0], CondClause::Arrow { .. }));
                assert!(matches!(clauses[1], CondClause::Else { .. }));
            }
            _ => panic!("expected cond"),
        }
    }

    #[test]
    fn parse_case() {
        let e = parse_ok("(case k ((1) a) (else b))");
        match e {
            Expr::Case { key, clauses } => {
                assert_eq!(*key, Expr::Var(id("k")));
                assert_eq!(clauses.len(), 2);
            }
            _ => panic!("expected case"),
        }
    }

    #[test]
    fn parse_let_named_and_plain() {
        let plain = parse_ok("(let ((x 1)) x)");
        match plain {
            Expr::Let { name: None, .. } => {}
            _ => panic!("expected plain let"),
        }
        let named = parse_ok("(let loop ((n 0)) n)");
        match named {
            Expr::Let {
                name: Some(n),
                ..
            } => assert_eq!(n, id("loop")),
            _ => panic!("expected named let"),
        }
    }

    #[test]
    fn parse_let_variants() {
        assert!(matches!(parse_ok("(let* ((x 1)) x)"), Expr::LetStar { .. }));
        assert!(matches!(parse_ok("(letrec ((x 1)) x)"), Expr::LetRec { .. }));
        assert!(matches!(parse_ok("(letrec* ((x 1)) x)"), Expr::LetRecStar { .. }));
    }

    #[test]
    fn parse_let_values() {
        let e = parse_ok("(let-values (((a b) (values 1 2))) a)");
        match e {
            Expr::LetValues { bindings, .. } => {
                assert_eq!(bindings.len(), 1);
                assert_eq!(bindings[0].formals.fixed, vec![id("a"), id("b")]);
            }
            _ => panic!("expected let-values"),
        }
        let e2 = parse_ok("(let*-values (((a) (values 1))) a)");
        assert!(matches!(e2, Expr::LetStarValues { .. }));
    }

    #[test]
    fn parse_begin_and_and_or() {
        assert!(matches!(parse_ok("(begin a b)"), Expr::Begin(v) if v.len() == 2));
        assert!(matches!(parse_ok("(and a b c)"), Expr::And(v) if v.len() == 3));
        assert!(matches!(parse_ok("(or a b)"), Expr::Or(v) if v.len() == 2));
    }

    #[test]
    fn parse_when_unless() {
        assert!(matches!(parse_ok("(when t a b)"), Expr::When { .. }));
        assert!(matches!(parse_ok("(unless t a)"), Expr::Unless { .. }));
    }

    #[test]
    fn parse_program_definitions_and_exprs() {
        let prog = parse_program("(define x 5) (define (f y) (+ y 1)) (f x)").unwrap();
        assert_eq!(prog.forms.len(), 3);
        assert!(matches!(prog.forms[0], TopLevelForm::Definition(Definition::Variable { .. })));
        assert!(matches!(prog.forms[1], TopLevelForm::Definition(Definition::Function { .. })));
        assert!(matches!(prog.forms[2], TopLevelForm::Expr(_)));
    }

    #[test]
    fn parse_define_syntax() {
        let prog = parse_program("(define-syntax kw (lambda (x) x))").unwrap();
        assert!(matches!(
            prog.forms[0],
            TopLevelForm::Definition(Definition::Syntax { .. })
        ));
    }

    #[test]
    fn parse_datum_list_and_vector_and_bytevector() {
        // used via quote so they become Literal datums
        assert!(matches!(parse_ok("'(1 . 2)"), Expr::Quote(_)));
        assert!(matches!(parse_ok("'#(1 2 3)"), Expr::Quote(Datum::Vector(_))));
        assert!(matches!(
            parse_ok("'#vu8(1 2 3)"),
            Expr::Quote(Datum::ByteVector(_))
        ));
    }

    #[test]
    fn parse_datum_comment_skips_next() {
        let e = parse_ok("(+ 1 #;2 3)");
        match e {
            Expr::Application { operands, .. } => assert_eq!(operands.len(), 2),
            _ => panic!("expected application"),
        }
    }

    #[test]
    fn parse_errors() {
        assert!(parse_expr("(").is_err());
        assert!(parse_expr("(lambda x)").is_err()); // no body
        assert!(parse_expr("(let ((x 1) 2) x)").is_err()); // malformed binding
        assert!(parse_expr("(+ 1 2").is_err()); // unclosed
    }

    fn list(items: &[Datum]) -> Datum {
        let mut d = Datum::Null;
        for it in items.iter().rev() {
            d = Datum::Pair(Box::new(it.clone()), Box::new(d));
        }
        d
    }

    // silence unused warning for helper
    #[test]
    fn list_helper_used() {
        let _ = list(&[]);
    }
}
