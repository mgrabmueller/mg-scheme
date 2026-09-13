//! Abstract syntax tree for the R6RS dialect of Scheme.
//!
//! The AST is a faithful representation of the surface syntax of the
//! `(rnrs base (6))` library plus the common forms defined in the R6RS
//! top-level program and library syntax (chapter 7). It is intentionally
//! *concrete* in shape: every syntactic form that has a distinct grammar
//! production gets its own node, so that a future macro-expansion or
//! desugaring pass can transform each form independently.
//!
//! The tree does not resolve binding structure or perform any semantic
//! analysis; that responsibility belongs to later passes.

use std::fmt;

/// A Scheme identifier: a variable or keyword reference.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident(pub String);

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A lexical datum, as defined by R6RS section 4.3.
///
/// Datums are the literal data carried by `quote`, `case` clauses, and the
/// read syntax. They are kept separate from [`Expr`] because a quoted
/// `(+ 1 2)` is data, not an expression to be evaluated.
#[derive(Debug, Clone, PartialEq)]
pub enum Datum {
    Boolean(bool),
    Character(char),
    Number(Number),
    String(String),
    Symbol(String),
    /// The empty list `()`.
    Null,
    /// A proper or improper list. The first element is the `car`; the
    /// second is the `cdr`, which is either another `Pair` or one of
    /// `Null`/`Datum` (for an improper tail).
    Pair(Box<Datum>, Box<Datum>),
    /// A vector `#(d ...)`.
    Vector(Vec<Datum>),
    /// A bytevector `#vu8(u8 ...)`.
    ByteVector(Vec<u8>),
}

/// R6RS numbers are split into the exactness tower of section 11.7.
///
/// Only the representable subsets are modeled here; the full numeric tower
/// (rationals, complexes) is layered on top of this in later passes.
#[derive(Debug, Clone, PartialEq)]
pub enum Number {
    /// An exact integer.
    Fixnum(i64),
    /// An inexact real (floating point).
    Flonum(f64),
}

/// Formal parameter list of a `lambda` expression (R6RS section 11.4.2).
///
/// The three permitted shapes are:
///   - fixed arity: `(a b c)`;
///   - fully variadic: `rest`;
///   - variadic tail: `(a b . rest)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formals {
    /// The fixed parameters, matched left to right.
    pub fixed: Vec<Ident>,
    /// The rest parameter, if any. `Some` means a trailing `. <var>` or a
    /// bare `<var>` form.
    pub rest: Option<Ident>,
}

/// A binding specification shared by the `let` family and `define`.
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: Ident,
    pub init: Box<Expr>,
}

/// A multiple-values binding, as used by `let-values` and `let*-values`.
/// The `formals` are matched against the values produced by `init`.
#[derive(Debug, Clone, PartialEq)]
pub struct MvBinding {
    pub formals: Formals,
    pub init: Box<Expr>,
}

/// The body of a `lambda` or binding form: zero or more internal
/// definitions followed by one or more expressions (R6RS section 11.3).
#[derive(Debug, Clone, PartialEq)]
pub struct Body {
    pub definitions: Vec<Definition>,
    pub expressions: Vec<Expr>,
}

/// An internal or top-level definition (R6RS section 11.2).
#[derive(Debug, Clone, PartialEq)]
pub enum Definition {
    /// `(define <var> <expr>)` or `(define <var>)`.
    Variable {
        name: Ident,
        init: Option<Box<Expr>>,
    },
    /// `(define (<var> . <formals>) <body>)`, the curried-definition shorthand.
    Function {
        name: Ident,
        formals: Formals,
        body: Body,
    },
    /// `(define-syntax <keyword> <expr>)`.
    Syntax {
        keyword: Ident,
        init: Box<Expr>,
    },
}

/// A `case` clause: either a datum-match clause or an `else` clause.
#[derive(Debug, Clone, PartialEq)]
pub enum CaseClause {
    /// `((d ...) <expr> ...)`.
    Matches {
        data: Vec<Datum>,
        body: Vec<Expr>,
    },
    /// `(else <expr> ...)`.
    Else {
        body: Vec<Expr>,
    },
}

/// A `cond` clause: a test plus a body, the `=>` form, or `else`.
#[derive(Debug, Clone, PartialEq)]
pub enum CondClause {
    /// `(<test> <expr> ...)`.
    Clause {
        test: Box<Expr>,
        body: Vec<Expr>,
    },
    /// `(<test> => <expr>)`.
    Arrow {
        test: Box<Expr>,
        recipient: Box<Expr>,
    },
    /// `(else <expr> ...)`.
    Else {
        body: Vec<Expr>,
    },
}

/// A Scheme expression (R6RS section 11.4).
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal datum quoted via `'`/`quote`.
    Literal(Datum),
    /// A variable reference.
    Var(Ident),
    /// `(quote <datum>)`.
    Quote(Datum),
    /// `(lambda <formals> <body>)`.
    Lambda {
        formals: Formals,
        body: Body,
    },
    /// `(if <test> <consequent> <alternate>?)`.
    If {
        test: Box<Expr>,
        consequent: Box<Expr>,
        alternate: Option<Box<Expr>>,
    },
    /// `(set! <var> <expr>)`.
    Set {
        var: Ident,
        value: Box<Expr>,
    },
    /// `(cond <cond-clause> ...)`.
    Cond(Vec<CondClause>),
    /// `(case <key> <case-clause> ...)`.
    Case {
        key: Box<Expr>,
        clauses: Vec<CaseClause>,
    },
    /// `(when <test> <expr> ...)`.
    When {
        test: Box<Expr>,
        body: Vec<Expr>,
    },
    /// `(unless <test> <expr> ...)`.
    Unless {
        test: Box<Expr>,
        body: Vec<Expr>,
    },
    /// `(and <expr> ...)`.
    And(Vec<Expr>),
    /// `(or <expr> ...)`.
    Or(Vec<Expr>),
    /// `(let <bindings> <body>)` or named let `(let <var> <bindings> <body>)`.
    Let {
        name: Option<Ident>,
        bindings: Vec<Binding>,
        body: Body,
    },
    /// `(let* <bindings> <body>)`.
    LetStar {
        bindings: Vec<Binding>,
        body: Body,
    },
    /// `(letrec <bindings> <body>)`.
    LetRec {
        bindings: Vec<Binding>,
        body: Body,
    },
    /// `(letrec* <bindings> <body>)`.
    LetRecStar {
        bindings: Vec<Binding>,
        body: Body,
    },
    /// `(let-values <mv-bindings> <body>)`.
    LetValues {
        bindings: Vec<MvBinding>,
        body: Body,
    },
    /// `(let*-values <mv-bindings> <body>)`.
    LetStarValues {
        bindings: Vec<MvBinding>,
        body: Body,
    },
    /// `(begin <expr> ...)`.
    Begin(Vec<Expr>),
    /// A procedure application: `(<operator> <operand> ...)`.
    Application {
        operator: Box<Expr>,
        operands: Vec<Expr>,
    },
}

/// A top-level program form (R6RS chapter 8 / library form of chapter 7).
#[derive(Debug, Clone, PartialEq)]
pub enum TopLevelForm {
    Definition(Definition),
    Expr(Expr),
    /// A `(begin <form> ...)` used purely for splicing at the top level.
    Begin(Vec<TopLevelForm>),
}

/// A complete R6RS program: a sequence of top-level forms.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub forms: Vec<TopLevelForm>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> Ident {
        Ident(s.to_string())
    }

    #[test]
    fn ident_display_and_eq() {
        assert_eq!(id("x").to_string(), "x");
        assert_eq!(id("x"), id("x"));
        assert_ne!(id("x"), id("y"));
    }

    #[test]
    fn empty_list_datum() {
        assert_eq!(Datum::Null, Datum::Null);
        assert_ne!(Datum::Null, Datum::Boolean(false));
    }

    #[test]
    fn fixed_formals() {
        let f = Formals {
            fixed: vec![id("a"), id("b"), id("c")],
            rest: None,
        };
        assert_eq!(f.fixed.len(), 3);
        assert!(f.rest.is_none());
    }

    #[test]
    fn variadic_formals() {
        let f = Formals {
            fixed: vec![id("a"), id("b")],
            rest: Some(id("rest")),
        };
        assert_eq!(f.fixed.len(), 2);
        assert_eq!(f.rest, Some(id("rest")));
    }

    #[test]
    fn fully_variadic_formals() {
        let f = Formals {
            fixed: vec![],
            rest: Some(id("args")),
        };
        assert!(f.fixed.is_empty());
        assert_eq!(f.rest, Some(id("args")));
    }

    #[test]
    fn lambda_expr() {
        let body = Body {
            definitions: vec![],
            expressions: vec![Expr::Var(id("x"))],
        };
        let lambda = Expr::Lambda {
            formals: Formals {
                fixed: vec![id("x")],
                rest: None,
            },
            body,
        };
        if let Expr::Lambda { formals, body } = lambda {
            assert_eq!(formals.fixed, vec![id("x")]);
            assert_eq!(body.expressions.len(), 1);
        } else {
            panic!("expected Lambda");
        }
    }

    #[test]
    fn if_without_alternate() {
        let iff = Expr::If {
            test: Box::new(Expr::Var(id("t"))),
            consequent: Box::new(Expr::Var(id("c"))),
            alternate: None,
        };
        match iff {
            Expr::If {
                alternate: None, ..
            } => {}
            _ => panic!("expected If with no alternate"),
        }
    }

    #[test]
    fn application_form() {
        let app = Expr::Application {
            operator: Box::new(Expr::Var(id("+"))),
            operands: vec![Expr::Literal(Datum::Number(Number::Fixnum(1)))],
        };
        match app {
            Expr::Application {
                operator,
                operands,
            } => {
                assert_eq!(*operator, Expr::Var(id("+")));
                assert_eq!(operands.len(), 1);
            }
            _ => panic!("expected Application"),
        }
    }

    #[test]
    fn named_let() {
        let expr = Expr::Let {
            name: Some(id("loop")),
            bindings: vec![Binding {
                name: id("n"),
                init: Box::new(Expr::Literal(Datum::Number(Number::Fixnum(0)))),
            }],
            body: Body {
                definitions: vec![],
                expressions: vec![],
            },
        };
        match expr {
            Expr::Let {
                name: Some(ref n),
                ..
            } => assert_eq!(*n, id("loop")),
            _ => panic!("expected named Let"),
        }
    }

    #[test]
    fn cond_arrow_clause() {
        let clause = CondClause::Arrow {
            test: Box::new(Expr::Var(id("x"))),
            recipient: Box::new(Expr::Var(id("f"))),
        };
        match clause {
            CondClause::Arrow { test, recipient } => {
                assert_eq!(*test, Expr::Var(id("x")));
                assert_eq!(*recipient, Expr::Var(id("f")));
            }
            _ => panic!("expected Arrow"),
        }
    }

    #[test]
    fn case_else_clause() {
        let clause = CaseClause::Else {
            body: vec![Expr::Var(id("default"))],
        };
        match clause {
            CaseClause::Else { body } => assert_eq!(body.len(), 1),
            _ => panic!("expected Else"),
        }
    }

    #[test]
    fn function_definition() {
        let def = Definition::Function {
            name: id("add3"),
            formals: Formals {
                fixed: vec![id("x")],
                rest: None,
            },
            body: Body {
                definitions: vec![],
                expressions: vec![Expr::Var(id("x"))],
            },
        };
        match def {
            Definition::Function { name, .. } => assert_eq!(name, id("add3")),
            _ => panic!("expected Function definition"),
        }
    }

    #[test]
    fn program_with_top_level_forms() {
        let prog = Program {
            forms: vec![
                TopLevelForm::Definition(Definition::Variable {
                    name: id("x"),
                    init: Some(Box::new(Expr::Literal(Datum::Number(Number::Fixnum(
                        42,
                    ))))),
                }),
                TopLevelForm::Expr(Expr::Var(id("x"))),
            ],
        };
        assert_eq!(prog.forms.len(), 2);
    }
}
