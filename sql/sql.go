// Package sql reads database/sql query results into dataframe Frames and
// writes Frames through prepared statements.
//
// The package deliberately leaves connections, transactions, query text,
// placeholders, and driver selection to database/sql. This makes it usable
// with any database/sql driver without imposing a SQL dialect.
package sql

import (
	"bytes"
	"context"
	stdsql "database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"time"

	"github.com/joeychilson/dataframe"
	"github.com/joeychilson/dataframe/internal/record"
	"github.com/joeychilson/dataframe/series"
)

// Queryer is implemented by *database/sql.DB, *database/sql.Tx, and
// *database/sql.Conn.
type Queryer interface {
	// QueryContext executes query with args and returns its rows.
	QueryContext(ctx context.Context, query string, args ...any) (*stdsql.Rows, error)
}

// Reader consumes database/sql rows as Frames or Go records. Create one with
// NewReader. A read always closes the rows, including when conversion fails.
type Reader struct {
	rows *stdsql.Rows
}

// NewReader returns a Reader that consumes rows.
func NewReader(rows *stdsql.Rows) *Reader {
	return &Reader{rows: rows}
}

// Read consumes and closes the query rows and returns a Frame. Result column
// names become frame column names. Values use database/sql's native scan types:
// bool, int64, float64, string, []byte, and time.Time. SQL NULL values create
// nullable columns. Empty and entirely-null columns fall back to
// ColumnType.ScanType metadata. Pointers are dereferenced; time.Time is
// preserved; and bool, integer, float, string, and byte-slice kinds normalize
// to the corresponding native scan type. A struct scan type is treated as a
// nullable wrapper only when it has exactly two exported fields: Valid bool and
// one recursively normalized value field. Nil, interface, and unrecognized
// scan types fall back to string. This fallback determines element type only;
// nullability still comes from ColumnType.Nullable or observed SQL NULL values.
// A nil Reader or nil rows returns an error.
func (r *Reader) Read() (frame dataframe.Frame, err error) {
	rows, err := r.takeRows()
	if err != nil {
		return dataframe.Frame{}, err
	}
	defer func() {
		if closeErr := rows.Close(); err == nil && closeErr != nil {
			err = closeErr
		}
	}()

	names, columnTypes, err := describeRows(rows)
	if err != nil {
		return dataframe.Frame{}, err
	}
	columns := make([]scannedColumn, len(names))
	for i, name := range names {
		columns[i].name = name
		if nullable, ok := columnTypes[i].Nullable(); ok {
			columns[i].nullable = nullable
		}
	}

	destinations := make([]any, len(columns))
	for i := range columns {
		destinations[i] = &columns[i]
	}
	row := 0
	for rows.Next() {
		if scanErr := rows.Scan(destinations...); scanErr != nil {
			return dataframe.Frame{}, fmt.Errorf("dataframe/sql: row %d: %w", row, scanErr)
		}
		row++
	}
	if rowsErr := rows.Err(); rowsErr != nil {
		return dataframe.Frame{}, rowsErr
	}
	if len(columns) == 0 && row > 0 {
		return dataframe.Frame{}, fmt.Errorf("%w: cannot represent %d sql rows without columns", dataframe.ErrUnsupported, row)
	}

	specs := make([]dataframe.ColumnSpec, len(columns))
	for i := range columns {
		if columns[i].typeOf == nil {
			columns[i].typeOf = normalizedScanType(columnTypes[i].ScanType())
			columns[i].appendZeroes(columns[i].length)
		}
		specs[i], err = columns[i].column()
		if err != nil {
			return dataframe.Frame{}, err
		}
	}
	return dataframe.New(specs...)
}

// ReadRecords consumes and closes the query rows into records of non-pointer
// struct type T. Fields use the same `df` tags as dataframe.FromRecords. Extra
// query columns are ignored. Pointer and series.Optional fields accept SQL
// NULL, and field types implementing database/sql.Scanner control conversion.
// A nil Reader or nil rows returns an error.
func (r *Reader) ReadRecords[T any]() (records []T, err error) {
	rows, err := r.takeRows()
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := rows.Close(); err == nil && closeErr != nil {
			err = closeErr
		}
	}()

	fields, err := record.Describe(reflect.TypeFor[T]())
	if err != nil {
		return nil, err
	}
	names, _, err := describeRows(rows)
	if err != nil {
		return nil, err
	}
	fieldByName := make(map[string]record.Field, len(fields))
	for _, field := range fields {
		fieldByName[field.Name] = field
	}
	columnFields := make([]record.Field, len(names))
	mapped := make([]bool, len(names))
	seenFields := make(map[string]struct{}, len(fields))
	for i, name := range names {
		field, ok := fieldByName[name]
		if !ok {
			continue
		}
		columnFields[i] = field
		mapped[i] = true
		seenFields[name] = struct{}{}
	}
	for _, field := range fields {
		if _, ok := seenFields[field.Name]; !ok {
			return nil, fmt.Errorf("%w: %q", dataframe.ErrColumnNotFound, field.Name)
		}
	}

	destinations := make([]any, len(names))
	scanners := make([]fieldScanner, len(names))
	for i := range destinations {
		if mapped[i] {
			scanners[i].field = columnFields[i]
			destinations[i] = &scanners[i]
		} else {
			destinations[i] = new(stdsql.RawBytes)
		}
	}
	var zero T
	for row := 0; rows.Next(); row++ {
		records = append(records, zero)
		recordValue := reflect.ValueOf(&records[len(records)-1]).Elem()
		for i := range scanners {
			if mapped[i] {
				scanners[i].record = recordValue
			}
		}
		if scanErr := rows.Scan(destinations...); scanErr != nil {
			return nil, fmt.Errorf("dataframe/sql: row %d: %w", row, scanErr)
		}
	}
	if rowsErr := rows.Err(); rowsErr != nil {
		return nil, rowsErr
	}
	if records == nil {
		return []T{}, nil
	}
	return records, nil
}

// Writer writes Frames or Go records through a prepared statement. Create one
// with NewWriter. Writer does not create transactions or prepare SQL; callers
// retain control of both through database/sql. It does not close the statement
// and may be reused.
type Writer struct {
	statement *stdsql.Stmt
}

// NewWriter returns a Writer that executes statement once per row.
func NewWriter(statement *stdsql.Stmt) *Writer {
	return &Writer{statement: statement}
}

// Write executes the prepared statement once per frame row. Arguments follow
// frame schema order, and null cells are passed as nil. Execution stops at the
// first error. Use a statement prepared on a transaction when atomicity is
// required. A nil Writer or statement returns an error.
func (w *Writer) Write(ctx context.Context, frame dataframe.Frame) error {
	if w == nil || w.statement == nil {
		return errors.New("dataframe/sql: nil statement")
	}
	statement := w.statement
	if frame.Width() == 0 && frame.Len() > 0 {
		return fmt.Errorf("%w: cannot execute %d rows without columns", dataframe.ErrUnsupported, frame.Len())
	}
	columns := slices.Collect(frame.Columns())
	arguments := make([]any, len(columns))
	for row := range frame.Len() {
		for i, column := range columns {
			value, present := column.At(row)
			if !present {
				arguments[i] = nil
				continue
			}
			arguments[i] = statementArgument(reflect.ValueOf(value))
		}
		if _, execErr := statement.ExecContext(ctx, arguments...); execErr != nil {
			return fmt.Errorf("dataframe/sql: row %d: %w", row, execErr)
		}
	}
	return nil
}

// WriteRecords executes the prepared statement once per record of non-pointer
// struct type T. Arguments follow record field order using the same `df` tags
// as dataframe.FromRecords. Nil pointers and absent series.Optional fields are
// passed as nil. Values implementing driver.Valuer control their SQL value. A
// nil Writer or statement returns an error.
func (w *Writer) WriteRecords[T any](ctx context.Context, records []T) error {
	if w == nil || w.statement == nil {
		return errors.New("dataframe/sql: nil statement")
	}
	statement := w.statement
	fields, err := record.Describe(reflect.TypeFor[T]())
	if err != nil {
		return err
	}
	if len(fields) == 0 && len(records) > 0 {
		return fmt.Errorf("%w: cannot execute %d records without fields", dataframe.ErrUnsupported, len(records))
	}

	values := reflect.ValueOf(records)
	arguments := make([]any, len(fields))
	for row := range records {
		value := values.Index(row)
		for i, field := range fields {
			fieldValue, present := field.Extract(value)
			if !present {
				arguments[i] = nil
				continue
			}
			arguments[i] = statementArgument(fieldValue)
		}
		if _, execErr := statement.ExecContext(ctx, arguments...); execErr != nil {
			return fmt.Errorf("dataframe/sql: row %d: %w", row, execErr)
		}
	}
	return nil
}

// Read consumes and closes rows using NewReader. Nil rows return an error.
func Read(rows *stdsql.Rows) (dataframe.Frame, error) {
	return NewReader(rows).Read()
}

// ReadRecords consumes and closes rows into records using NewReader. Nil rows
// return an error.
func ReadRecords[T any](rows *stdsql.Rows) ([]T, error) {
	return NewReader(rows).ReadRecords[T]()
}

// Query runs query through q, consumes and closes its rows, and returns a
// Frame. q may be a *database/sql.DB, *database/sql.Tx, or *database/sql.Conn.
// A nil q returns an error.
func Query(ctx context.Context, q Queryer, query string, args ...any) (dataframe.Frame, error) {
	if isNilInterfaceValue(q) {
		return dataframe.Frame{}, errors.New("dataframe/sql: nil queryer")
	}
	rows, err := q.QueryContext(ctx, query, args...)
	if err != nil {
		return dataframe.Frame{}, err
	}
	return Read(rows)
}

// QueryRecords runs query through q, consumes and closes its rows, and returns
// records of non-pointer struct type T. A nil q returns an error.
func QueryRecords[T any](ctx context.Context, q Queryer, query string, args ...any) ([]T, error) {
	if isNilInterfaceValue(q) {
		return nil, errors.New("dataframe/sql: nil queryer")
	}
	rows, err := q.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	return ReadRecords[T](rows)
}

// Write executes statement once per frame row using NewWriter. A nil statement
// returns an error.
func Write(ctx context.Context, statement *stdsql.Stmt, frame dataframe.Frame) error {
	return NewWriter(statement).Write(ctx, frame)
}

// Keep temporary column blocks bounded while avoiding repeated copying from
// geometrically growing one large slice.
const sqlColumnTargetBlockBytes = 16 << 10

func (r *Reader) takeRows() (*stdsql.Rows, error) {
	if r == nil || r.rows == nil {
		return nil, errors.New("dataframe/sql: nil rows")
	}
	rows := r.rows
	r.rows = nil
	return rows, nil
}

func describeRows(rows *stdsql.Rows) ([]string, []*stdsql.ColumnType, error) {
	names, err := rows.Columns()
	if err != nil {
		return nil, nil, err
	}
	types, err := rows.ColumnTypes()
	if err != nil {
		return nil, nil, err
	}
	if len(types) != len(names) {
		return nil, nil, fmt.Errorf("%w: database/sql returned %d column types for %d columns", dataframe.ErrSchemaMismatch, len(types), len(names))
	}
	seen := make(map[string]struct{}, len(names))
	for i, name := range names {
		if name == "" {
			return nil, nil, fmt.Errorf("%w: sql column %d", dataframe.ErrInvalidName, i)
		}
		if _, ok := seen[name]; ok {
			return nil, nil, fmt.Errorf("%w: sql column %q", dataframe.ErrColumnConflict, name)
		}
		seen[name] = struct{}{}
	}
	return names, types, nil
}

type scannedColumn struct {
	name     string
	typeOf   reflect.Type
	nullable bool
	length   int
	validity []bool

	bools   scannedValues[bool]
	ints    scannedValues[int64]
	floats  scannedValues[float64]
	strings scannedValues[string]
	bytes   scannedValues[[]byte]
	times   scannedValues[time.Time]
}

type scannedValues[T any] struct {
	blocks    [][]T
	blockRows int
}

func (v *scannedValues[T]) append(value T) {
	if len(v.blocks) == 0 || len(v.blocks[len(v.blocks)-1]) == cap(v.blocks[len(v.blocks)-1]) {
		v.grow()
	}
	block := len(v.blocks) - 1
	v.blocks[block] = append(v.blocks[block], value)
}

func (v *scannedValues[T]) appendZeroes(count int) {
	for count > 0 {
		if len(v.blocks) == 0 || len(v.blocks[len(v.blocks)-1]) == cap(v.blocks[len(v.blocks)-1]) {
			v.grow()
		}
		block := len(v.blocks) - 1
		available := cap(v.blocks[block]) - len(v.blocks[block])
		added := min(count, available)
		v.blocks[block] = v.blocks[block][:len(v.blocks[block])+added]
		count -= added
	}
}

func (v *scannedValues[T]) grow() {
	if v.blockRows == 0 {
		size := max(uintptr(1), reflect.TypeFor[T]().Size())
		v.blockRows = max(1, int(uintptr(sqlColumnTargetBlockBytes)/size))
	}
	v.blocks = append(v.blocks, make([]T, 0, v.blockRows))
}

func (v scannedValues[T]) at(row int) T {
	return v.blocks[row/v.blockRows][row%v.blockRows]
}

// Scan implements database/sql.Scanner. database/sql only guarantees that a
// []byte source remains valid until the next Scan call, so byte values are
// cloned as they enter the column builder.
func (c *scannedColumn) Scan(value any) error {
	if value == nil {
		c.appendZeroes(1)
		c.appendValidity(false)
		c.nullable = true
		c.length++
		return nil
	}
	typeOf := reflect.TypeOf(value)
	if c.typeOf == nil {
		c.typeOf = typeOf
		c.appendZeroes(c.length)
	} else if c.typeOf != typeOf {
		return fmt.Errorf("%w: column %q changed from %v to %v at row %d", dataframe.ErrColumnType, c.name, c.typeOf, typeOf, c.length)
	}
	switch value := value.(type) {
	case bool:
		c.bools.append(value)
	case int64:
		c.ints.append(value)
	case float64:
		c.floats.append(value)
	case string:
		c.strings.append(value)
	case []byte:
		c.bytes.append(bytes.Clone(value))
	case time.Time:
		c.times.append(value)
	}
	c.appendValidity(true)
	c.length++
	return nil
}

func (c *scannedColumn) appendZeroes(count int) {
	switch c.typeOf {
	case reflect.TypeFor[bool]():
		c.bools.appendZeroes(count)
	case reflect.TypeFor[int64]():
		c.ints.appendZeroes(count)
	case reflect.TypeFor[float64]():
		c.floats.appendZeroes(count)
	case reflect.TypeFor[string]():
		c.strings.appendZeroes(count)
	case reflect.TypeFor[[]byte]():
		c.bytes.appendZeroes(count)
	case reflect.TypeFor[time.Time]():
		c.times.appendZeroes(count)
	}
}

func (c *scannedColumn) appendValidity(present bool) {
	if c.validity != nil {
		c.validity = append(c.validity, present)
		return
	}
	if present {
		return
	}
	c.validity = make([]bool, c.length, c.length+1)
	for row := range c.validity {
		c.validity[row] = true
	}
	c.validity = append(c.validity, false)
}

func (c scannedColumn) column() (dataframe.ColumnSpec, error) {
	switch c.typeOf {
	case reflect.TypeFor[bool]():
		return scannedColumnOf(c, c.bools), nil
	case reflect.TypeFor[int64]():
		return scannedColumnOf(c, c.ints), nil
	case reflect.TypeFor[float64]():
		return scannedColumnOf(c, c.floats), nil
	case reflect.TypeFor[string]():
		return scannedColumnOf(c, c.strings), nil
	case reflect.TypeFor[[]byte]():
		return scannedColumnOf(c, c.bytes), nil
	case reflect.TypeFor[time.Time]():
		return scannedColumnOf(c, c.times), nil
	default:
		return nil, fmt.Errorf("%w: sql column %q has scan type %v", dataframe.ErrUnsupported, c.name, c.typeOf)
	}
}

func scannedColumnOf[T any](column scannedColumn, values scannedValues[T]) dataframe.ColumnSpec {
	if !column.nullable {
		result := series.NewFunc(column.length, values.at)
		return dataframe.ColumnFromSeries(column.name, result)
	}
	result := series.NewNullableFunc(column.length, func(row int) (T, bool) {
		present := column.validity == nil || column.validity[row]
		if !present {
			var zero T
			return zero, false
		}
		return values.at(row), true
	})
	return dataframe.ColumnFromSeries(column.name, result)
}

func normalizedScanType(typeOf reflect.Type) reflect.Type {
	for typeOf != nil && typeOf.Kind() == reflect.Pointer {
		typeOf = typeOf.Elem()
	}
	if typeOf == nil || typeOf.Kind() == reflect.Interface {
		return reflect.TypeFor[string]()
	}
	if typeOf == reflect.TypeFor[time.Time]() {
		return typeOf
	}
	if valueType, ok := nullableValueType(typeOf); ok {
		return normalizedScanType(valueType)
	}
	switch typeOf.Kind() {
	case reflect.Bool:
		return reflect.TypeFor[bool]()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return reflect.TypeFor[int64]()
	case reflect.Float32, reflect.Float64:
		return reflect.TypeFor[float64]()
	case reflect.String:
		return reflect.TypeFor[string]()
	case reflect.Slice:
		if typeOf.Elem().Kind() == reflect.Uint8 {
			return reflect.TypeFor[[]byte]()
		}
	}
	return reflect.TypeFor[string]()
}

func nullableValueType(typeOf reflect.Type) (reflect.Type, bool) {
	if typeOf.Kind() != reflect.Struct || typeOf.NumField() != 2 {
		return nil, false
	}
	var valueType reflect.Type
	valid := false
	for field := range typeOf.Fields() {
		if field.PkgPath != "" {
			return nil, false
		}
		if field.Name == "Valid" {
			if field.Type != reflect.TypeFor[bool]() {
				return nil, false
			}
			valid = true
		} else {
			valueType = field.Type
		}
	}
	return valueType, valid && valueType != nil
}

type fieldScanner struct {
	field  record.Field
	record reflect.Value
}

// Scan converts a database value into the record field destination.
func (s *fieldScanner) Scan(source any) error {
	if source == nil {
		if s.field.Nullable() {
			return nil
		}
		destination := s.field.Destination(s.record)
		if destination.Addr().Type().Implements(reflect.TypeFor[stdsql.Scanner]()) {
			return stdsql.ConvertAssign(driver.ScanContext{}, destination.Addr().Interface(), source)
		}
		return fmt.Errorf("%w: null in non-null field %s", dataframe.ErrInvalidRecord, s.field.Name)
	}
	destination := s.field.Destination(s.record)
	return stdsql.ConvertAssign(driver.ScanContext{}, destination.Addr().Interface(), source)
}

func statementArgument(value reflect.Value) any {
	if !value.IsValid() {
		return nil
	}
	for value.Kind() == reflect.Interface {
		if value.IsNil() {
			return nil
		}
		value = value.Elem()
	}
	valuer := reflect.TypeFor[driver.Valuer]()
	if value.Type().Implements(valuer) {
		return value.Interface()
	}
	pointerType := reflect.PointerTo(value.Type())
	if pointerType.Implements(valuer) {
		pointer := reflect.New(value.Type())
		pointer.Elem().Set(value)
		return pointer.Interface()
	}
	return value.Interface()
}

func isNilInterfaceValue(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}
