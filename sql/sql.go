package sql

import (
	"context"
	stdsql "database/sql"
	"database/sql/driver"
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
	QueryContext(context.Context, string, ...any) (*stdsql.Rows, error)
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
// nullable columns. Empty and entirely-null results use ColumnType.ScanType
// when it identifies one of those types, and otherwise use string.
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

	values := make([]any, len(columns))
	destinations := make([]any, len(columns))
	for i := range values {
		destinations[i] = &values[i]
	}
	row := 0
	for rows.Next() {
		if err := rows.Scan(destinations...); err != nil {
			return dataframe.Frame{}, fmt.Errorf("dataframe/sql: row %d: %w", row, err)
		}
		for i, value := range values {
			if err := columns[i].append(value, row); err != nil {
				return dataframe.Frame{}, err
			}
		}
		row++
	}
	if err := rows.Err(); err != nil {
		return dataframe.Frame{}, err
	}
	if len(columns) == 0 && row > 0 {
		return dataframe.Frame{}, fmt.Errorf("%w: cannot represent %d SQL rows without columns", dataframe.ErrUnsupported, row)
	}

	specs := make([]dataframe.ColumnSpec, len(columns))
	for i := range columns {
		if columns[i].typeOf == nil {
			columns[i].typeOf = normalizedScanType(columnTypes[i].ScanType())
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

	fields, err := record.Describe(reflect.TypeFor[T](), dataframe.ErrInvalidRecord, dataframe.ErrInvalidName, dataframe.ErrColumnConflict)
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
			destinations[i] = new(any)
		}
	}
	for row := 0; rows.Next(); row++ {
		recordValue := reflect.New(reflect.TypeFor[T]()).Elem()
		for i := range scanners {
			if mapped[i] {
				scanners[i].record = recordValue
			}
		}
		if err := rows.Scan(destinations...); err != nil {
			return nil, fmt.Errorf("dataframe/sql: row %d: %w", row, err)
		}
		records = append(records, recordValue.Interface().(T))
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if records == nil {
		return []T{}, nil
	}
	return records, nil
}

func (r *Reader) takeRows() (*stdsql.Rows, error) {
	if r == nil || r.rows == nil {
		return nil, fmt.Errorf("dataframe/sql: nil rows")
	}
	rows := r.rows
	r.rows = nil
	return rows, nil
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
// required.
func (w *Writer) Write(ctx context.Context, frame dataframe.Frame) error {
	statement, err := w.configuredStatement()
	if err != nil {
		return err
	}
	if frame.Width() == 0 && frame.Len() > 0 {
		return fmt.Errorf("%w: cannot execute %d rows without columns", dataframe.ErrUnsupported, frame.Len())
	}
	columns := slices.Collect(frame.Columns())
	arguments := make([]any, len(columns))
	for row := 0; row < frame.Len(); row++ {
		for i, column := range columns {
			value, present := column.At(row)
			if !present {
				arguments[i] = nil
				continue
			}
			arguments[i] = statementArgument(reflect.ValueOf(value))
		}
		if _, err := statement.ExecContext(ctx, arguments...); err != nil {
			return fmt.Errorf("dataframe/sql: row %d: %w", row, err)
		}
	}
	return nil
}

// WriteRecords executes the prepared statement once per record of non-pointer
// struct type T. Arguments follow record field order using the same `df` tags
// as dataframe.FromRecords. Nil pointers and absent series.Optional fields are
// passed as nil. Values implementing driver.Valuer control their SQL value.
func (w *Writer) WriteRecords[T any](ctx context.Context, records []T) error {
	statement, err := w.configuredStatement()
	if err != nil {
		return err
	}
	fields, err := record.Describe(reflect.TypeFor[T](), dataframe.ErrInvalidRecord, dataframe.ErrInvalidName, dataframe.ErrColumnConflict)
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
		if _, err := statement.ExecContext(ctx, arguments...); err != nil {
			return fmt.Errorf("dataframe/sql: row %d: %w", row, err)
		}
	}
	return nil
}

func (w *Writer) configuredStatement() (*stdsql.Stmt, error) {
	if w == nil || w.statement == nil {
		return nil, fmt.Errorf("dataframe/sql: nil statement")
	}
	return w.statement, nil
}

// Read consumes and closes rows using NewReader.
func Read(rows *stdsql.Rows) (dataframe.Frame, error) {
	return NewReader(rows).Read()
}

// ReadRecords consumes and closes rows into records using NewReader.
func ReadRecords[T any](rows *stdsql.Rows) ([]T, error) {
	return NewReader(rows).ReadRecords[T]()
}

// Query runs query through q, consumes and closes its rows, and returns a
// Frame. q may be a *database/sql.DB, *database/sql.Tx, or *database/sql.Conn.
func Query(ctx context.Context, q Queryer, query string, args ...any) (dataframe.Frame, error) {
	if isNil(q) {
		return dataframe.Frame{}, fmt.Errorf("dataframe/sql: nil queryer")
	}
	rows, err := q.QueryContext(ctx, query, args...)
	if err != nil {
		return dataframe.Frame{}, err
	}
	return Read(rows)
}

// QueryRecords runs query through q, consumes and closes its rows, and returns
// records of non-pointer struct type T.
func QueryRecords[T any](ctx context.Context, q Queryer, query string, args ...any) ([]T, error) {
	if isNil(q) {
		return nil, fmt.Errorf("dataframe/sql: nil queryer")
	}
	rows, err := q.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	return ReadRecords[T](rows)
}

// Write executes statement once per frame row using NewWriter.
func Write(ctx context.Context, statement *stdsql.Stmt, frame dataframe.Frame) error {
	return NewWriter(statement).Write(ctx, frame)
}

type scannedColumn struct {
	name     string
	values   []any
	typeOf   reflect.Type
	nullable bool
}

func (c *scannedColumn) append(value any, row int) error {
	if value == nil {
		c.values = append(c.values, nil)
		c.nullable = true
		return nil
	}
	typeOf := reflect.TypeOf(value)
	if c.typeOf == nil {
		c.typeOf = typeOf
	} else if c.typeOf != typeOf {
		return fmt.Errorf("%w: column %q changed from %v to %v at row %d", dataframe.ErrColumnType, c.name, c.typeOf, typeOf, row)
	}
	c.values = append(c.values, value)
	return nil
}

func (c scannedColumn) column() (dataframe.ColumnSpec, error) {
	switch c.typeOf {
	case reflect.TypeFor[bool]():
		return scannedColumnOf[bool](c)
	case reflect.TypeFor[int64]():
		return scannedColumnOf[int64](c)
	case reflect.TypeFor[float64]():
		return scannedColumnOf[float64](c)
	case reflect.TypeFor[string]():
		return scannedColumnOf[string](c)
	case reflect.TypeFor[[]byte]():
		return scannedColumnOf[[]byte](c)
	case reflect.TypeFor[time.Time]():
		return scannedColumnOf[time.Time](c)
	default:
		return nil, fmt.Errorf("%w: SQL column %q has scan type %v", dataframe.ErrUnsupported, c.name, c.typeOf)
	}
}

func scannedColumnOf[T any](column scannedColumn) (dataframe.ColumnSpec, error) {
	values := make([]T, len(column.values))
	var validity []bool
	if column.nullable {
		validity = make([]bool, len(values))
	}
	for i, value := range column.values {
		if value == nil {
			continue
		}
		typed, ok := value.(T)
		if !ok {
			return nil, fmt.Errorf("%w: SQL column %q contains %T, want %v", dataframe.ErrColumnType, column.name, value, reflect.TypeFor[T]())
		}
		values[i] = typed
		if validity != nil {
			validity[i] = true
		}
	}
	if validity == nil {
		return dataframe.Column(column.name, values), nil
	}
	nullable, err := series.NewNullable(values, validity)
	if err != nil {
		panic(err)
	}
	return dataframe.ColumnFromSeries(column.name, nullable), nil
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
			return nil, nil, fmt.Errorf("%w: SQL column %d", dataframe.ErrInvalidName, i)
		}
		if _, ok := seen[name]; ok {
			return nil, nil, fmt.Errorf("%w: SQL column %q", dataframe.ErrColumnConflict, name)
		}
		seen[name] = struct{}{}
	}
	return names, types, nil
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
	if typeOf.Kind() != reflect.Struct {
		return nil, false
	}
	valid, ok := typeOf.FieldByName("Valid")
	if !ok || valid.Type != reflect.TypeFor[bool]() {
		return nil, false
	}
	for field := range typeOf.Fields() {
		if field.Name != "Valid" && field.PkgPath == "" {
			return field.Type, true
		}
	}
	return nil, false
}

type fieldScanner struct {
	field  record.Field
	record reflect.Value
}

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

func isNil(value any) bool {
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
