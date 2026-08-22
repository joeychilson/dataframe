// Package sql reads database/sql query results into dataframe Frames and
// writes Frames through prepared statements.
//
// The package deliberately leaves connections, transactions, query text,
// placeholders, and driver selection to database/sql. This makes it usable
// with any database/sql driver without imposing a SQL dialect.
package sql
