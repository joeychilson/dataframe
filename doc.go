// Package dataframe provides immutable, ordered tables whose columns are
// typed series values.
//
// A Frame's schema is dynamic because column names are strings. [Frame.Column]
// is the checked boundary: it verifies the name and exact Go type, then returns
// an ordinary typed Series. [Frame.Columns] provides read-only dynamic access
// for schema-driven code. Series and masks align positionally; operations
// validate lengths whenever they combine rows.
//
// Panics are reserved for programmer-contract violations in operations whose
// signatures cannot return errors, including invalid indexes or bounds,
// incompatible positional lengths, and invalid callback or numeric arguments.
// Dynamic schema operations, construction, joins, and record conversion return
// errors.
package dataframe

import "errors"

var (
	// ErrColumnNotFound reports that a requested column name is absent.
	ErrColumnNotFound = errors.New("column not found")
	// ErrColumnType reports that a column does not have the requested exact Go type.
	ErrColumnType = errors.New("column type mismatch")
	// ErrRowCount reports incompatible positional lengths or frame row counts.
	ErrRowCount = errors.New("row count mismatch")
	// ErrInvalidName reports an empty column name.
	ErrInvalidName = errors.New("column name must not be empty")
	// ErrColumnConflict reports duplicate or colliding output column names.
	ErrColumnConflict = errors.New("column name conflict")
	// ErrSchemaMismatch reports frames whose ordered schemas cannot be combined.
	ErrSchemaMismatch = errors.New("frame schema mismatch")
	// ErrUnsupported reports a value or type that an operation cannot handle.
	ErrUnsupported = errors.New("unsupported operation for column type")
	// ErrInvalidRecord reports an unsupported record type or field value.
	ErrInvalidRecord = errors.New("invalid record")
	// ErrInvalidRow reports an unusable row value.
	ErrInvalidRow = errors.New("invalid row")
)
