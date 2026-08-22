package csv

import (
	"encoding"
	stdcsv "encoding/csv"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/joeychilson/dataframe"
	"github.com/joeychilson/dataframe/internal/record"
	"github.com/joeychilson/dataframe/series"
)

// Reader reads CSV data as Frames or Go records. Create one with NewReader.
type Reader struct {
	// Comma is the field delimiter. NewReader sets it to ','.
	Comma rune

	// Comment, when non-zero, starts comment lines. It follows encoding/csv:
	// the comment character must be a valid rune distinct from Comma and may
	// not be '\r', '\n', or the Unicode replacement character.
	Comment rune

	// FieldsPerRecord controls the parser's expected record width, following
	// encoding/csv.Reader. Zero uses the first record's width; a positive value
	// requires exactly that many fields; a negative value disables the parser's
	// width check. Read still rejects ragged data because every Frame row must
	// have the same width.
	FieldsPerRecord int

	// LazyQuotes permits quotes in unquoted fields and non-doubled quotes in
	// quoted fields, following encoding/csv.Reader.
	LazyQuotes bool

	// TrimLeadingSpace ignores leading space in fields, following
	// encoding/csv.Reader.
	TrimLeadingSpace bool

	// Header reports whether the first non-comment record contains column names.
	// NewReader sets it to true. When false, Read synthesizes names column1,
	// column2, and so on.
	Header bool

	// NullValues contains exact tokens interpreted as null. NewReader initializes
	// it to []string{""}. A nil or empty slice disables null recognition. Null
	// tokens do not participate in type inference.
	NullValues []string

	// InferRows limits type inference to the first n data rows. Zero examines
	// the complete input. A negative value causes Read to return an error.
	InferRows int

	input io.Reader
}

// NewReader returns a Reader using the package defaults.
func NewReader(r io.Reader) *Reader {
	return &Reader{
		Comma:      ',',
		Header:     true,
		NullValues: []string{""},
		input:      r,
	}
}

// Read parses CSV into an inferred Frame. Non-null columns use the ladder bool,
// int64, float64, string. Boolean inference recognizes textual true and false,
// not numeric 0 and 1. An empty input produces the zero Frame. Header-only
// columns have non-null string type. Entirely-null columns have nullable string
// type. Ragged records, duplicate or empty headers, and values incompatible
// with a sampled type return errors.
func (r *Reader) Read() (dataframe.Frame, error) {
	if r != nil && r.InferRows < 0 {
		return dataframe.Frame{}, fmt.Errorf("csv: InferRows must not be negative")
	}
	input, err := r.readInput()
	if err != nil {
		return dataframe.Frame{}, err
	}
	if len(input.names) == 0 {
		return dataframe.Frame{}, nil
	}
	nulls := makeStringSet(r.NullValues)
	sampleRows := len(input.rows)
	if r.InferRows > 0 {
		sampleRows = min(sampleRows, r.InferRows)
	}
	columns := make([]dataframe.ColumnSpec, len(input.names))
	for columnIndex, name := range input.names {
		kind := inferString
		seen := false
		boolPossible := true
		intPossible := true
		floatPossible := true
		for row := 0; row < sampleRows; row++ {
			text := input.rows[row][columnIndex]
			if _, null := nulls[text]; null {
				continue
			}
			seen = true
			if boolPossible {
				_, boolPossible = parseBool(text)
			}
			if intPossible {
				_, intErr := strconv.ParseInt(text, 10, 64)
				intPossible = intErr == nil
			}
			if floatPossible {
				_, floatErr := strconv.ParseFloat(text, 64)
				floatPossible = floatErr == nil
			}
			if !boolPossible && !intPossible && !floatPossible {
				break
			}
		}
		if seen {
			switch {
			case boolPossible:
				kind = inferBool
			case intPossible:
				kind = inferInt
			case floatPossible:
				kind = inferFloat
			}
		}

		switch kind {
		case inferBool:
			columns[columnIndex], err = parsedColumn(name, input.rows, columnIndex, nulls, func(text string) (bool, error) {
				value, ok := parseBool(text)
				if !ok {
					return false, fmt.Errorf("invalid boolean %q", text)
				}
				return value, nil
			})
		case inferInt:
			columns[columnIndex], err = parsedColumn(name, input.rows, columnIndex, nulls, func(text string) (int64, error) {
				return strconv.ParseInt(text, 10, 64)
			})
		case inferFloat:
			columns[columnIndex], err = parsedColumn(name, input.rows, columnIndex, nulls, func(text string) (float64, error) {
				return strconv.ParseFloat(text, 64)
			})
		case inferString:
			columns[columnIndex], err = parsedColumn(name, input.rows, columnIndex, nulls, func(text string) (string, error) {
				return text, nil
			})
		}
		if err != nil {
			return dataframe.Frame{}, err
		}
	}
	return dataframe.New(columns...)
}

// ReadRecords parses CSV into records of non-pointer struct type T using the
// same `df` tags as dataframe.FromRecords. It uses the struct field types
// directly and performs no type inference. Pointer and series.Optional fields
// are nullable. Scalar fields implementing encoding.TextUnmarshaler control
// their parsing.
func (r *Reader) ReadRecords[T any]() ([]T, error) {
	typeOf := reflect.TypeFor[T]()
	fields, err := record.Describe(typeOf, dataframe.ErrInvalidRecord, dataframe.ErrInvalidName, dataframe.ErrColumnConflict)
	if err != nil {
		return nil, err
	}
	input, err := r.readInput()
	if err != nil {
		return nil, err
	}
	if len(input.names) == 0 && len(input.rows) == 0 {
		return []T{}, nil
	}

	indexes := make(map[string]int, len(input.names))
	for i, name := range input.names {
		indexes[name] = i
	}
	fieldColumns := make([]int, len(fields))
	for i, field := range fields {
		index, found := indexes[field.Name]
		if !found {
			return nil, fmt.Errorf("%w: %q", dataframe.ErrColumnNotFound, field.Name)
		}
		fieldColumns[i] = index
	}

	nulls := makeStringSet(r.NullValues)
	records := make([]T, len(input.rows))
	for row := range records {
		recordValue := reflect.ValueOf(&records[row]).Elem()
		for i, field := range fields {
			text := input.rows[row][fieldColumns[i]]
			_, null := nulls[text]
			if null {
				if !field.Nullable() {
					return nil, fmt.Errorf("%w: null in non-null field %s at row %d", dataframe.ErrInvalidRecord, field.Name, row)
				}
				continue
			}
			destination := field.Destination(recordValue)
			if err := unmarshalValue(destination, text); err != nil {
				return nil, fmt.Errorf("csv: row %d column %q: %w", row, field.Name, err)
			}
		}
	}
	return records, nil
}

// Writer writes Frames or Go records as CSV. Create one with NewWriter.
type Writer struct {
	// Comma is the field delimiter. NewWriter sets it to ','.
	Comma rune

	// UseCRLF writes \r\n line endings instead of \n, following
	// encoding/csv.Writer.
	UseCRLF bool

	// Header reports whether to write column names. NewWriter sets it to true.
	Header bool

	// NullString is the token written for null cells. NewWriter sets it to the
	// empty string.
	NullString string

	output io.Writer
}

// NewWriter returns a Writer using the package defaults.
func NewWriter(w io.Writer) *Writer {
	return &Writer{Comma: ',', Header: true, output: w}
}

// Write serializes f to CSV. Values implementing encoding.TextMarshaler
// control their text form. Unsupported element types return an error wrapping
// dataframe.ErrUnsupported.
func (w *Writer) Write(f dataframe.Frame) error {
	writer, err := w.configuredWriter()
	if err != nil {
		return err
	}
	if f.Width() == 0 && f.Len() > 0 {
		return fmt.Errorf("%w: cannot encode %d rows without columns", dataframe.ErrUnsupported, f.Len())
	}
	if w.Header {
		if err := writer.Write(f.Names()); err != nil {
			return err
		}
	}
	columns := slices.Collect(f.Columns())
	record := make([]string, len(columns))
	for row := 0; row < f.Len(); row++ {
		for columnIndex, column := range columns {
			value, present := column.At(row)
			if !present {
				record[columnIndex] = w.NullString
				continue
			}
			text, err := marshalValue(value)
			if err != nil {
				return fmt.Errorf("csv: row %d column %q: %w", row, column.Name(), err)
			}
			record[columnIndex] = text
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	writer.Flush()
	return writer.Error()
}

// WriteRecords writes records of non-pointer struct type T using the same `df`
// tags as dataframe.FromRecords. Pointer and absent series.Optional fields
// write NullString. Values implementing encoding.TextMarshaler control their
// text form.
func (w *Writer) WriteRecords[T any](records []T) error {
	typeOf := reflect.TypeFor[T]()
	fields, err := record.Describe(typeOf, dataframe.ErrInvalidRecord, dataframe.ErrInvalidName, dataframe.ErrColumnConflict)
	if err != nil {
		return err
	}
	writer, err := w.configuredWriter()
	if err != nil {
		return err
	}
	if len(fields) == 0 && len(records) > 0 {
		return fmt.Errorf("%w: cannot encode %d records without fields", dataframe.ErrUnsupported, len(records))
	}
	if w.Header {
		header := make([]string, len(fields))
		for i, field := range fields {
			header[i] = field.Name
		}
		if err := writer.Write(header); err != nil {
			return err
		}
	}

	values := reflect.ValueOf(records)
	encoded := make([]string, len(fields))
	for row := range records {
		value := values.Index(row)
		for fieldIndex, field := range fields {
			fieldValue, present := field.Extract(value)
			if !present {
				encoded[fieldIndex] = w.NullString
				continue
			}
			text, err := marshalReflectValue(fieldValue)
			if err != nil {
				return fmt.Errorf("csv: row %d column %q: %w", row, field.Name, err)
			}
			encoded[fieldIndex] = text
		}
		if err := writer.Write(encoded); err != nil {
			return err
		}
	}
	writer.Flush()
	return writer.Error()
}

// Read parses CSV using NewReader's defaults.
func Read(r io.Reader) (dataframe.Frame, error) {
	return NewReader(r).Read()
}

// Write serializes f using NewWriter's defaults.
func Write(w io.Writer, f dataframe.Frame) error {
	return NewWriter(w).Write(f)
}

func (w *Writer) configuredWriter() (*stdcsv.Writer, error) {
	if w == nil || w.output == nil {
		return nil, fmt.Errorf("csv: nil writer")
	}
	if !validDelimiter(w.Comma) {
		return nil, fmt.Errorf("csv: invalid field or comment delimiter")
	}
	writer := stdcsv.NewWriter(w.output)
	writer.Comma = w.Comma
	writer.UseCRLF = w.UseCRLF
	return writer, nil
}

type inferredKind uint8

const (
	inferString inferredKind = iota
	inferBool
	inferInt
	inferFloat
)

type csvInput struct {
	names []string
	rows  [][]string
}

func (r *Reader) readInput() (csvInput, error) {
	if r == nil || r.input == nil {
		return csvInput{}, fmt.Errorf("csv: nil reader")
	}
	reader := stdcsv.NewReader(r.input)
	reader.Comma = r.Comma
	reader.Comment = r.Comment
	reader.FieldsPerRecord = r.FieldsPerRecord
	reader.LazyQuotes = r.LazyQuotes
	reader.TrimLeadingSpace = r.TrimLeadingSpace
	records, err := reader.ReadAll()
	if err != nil {
		return csvInput{}, err
	}
	if len(records) == 0 {
		return csvInput{}, nil
	}

	result := csvInput{}
	if r.Header {
		result.names = slices.Clone(records[0])
		result.rows = records[1:]
	} else {
		result.rows = records
		result.names = make([]string, len(records[0]))
		for i := range result.names {
			result.names[i] = fmt.Sprintf("column%d", i+1)
		}
	}
	names := make(map[string]struct{}, len(result.names))
	for i, name := range result.names {
		if name == "" {
			return csvInput{}, fmt.Errorf("%w: header column %d", dataframe.ErrInvalidName, i)
		}
		if _, exists := names[name]; exists {
			return csvInput{}, fmt.Errorf("%w: header %q", dataframe.ErrColumnConflict, name)
		}
		names[name] = struct{}{}
	}
	for row, values := range result.rows {
		if len(values) != len(result.names) {
			return csvInput{}, fmt.Errorf("%w: CSV row %d has %d fields, want %d", dataframe.ErrRowCount, row, len(values), len(result.names))
		}
	}
	return result, nil
}

func parsedColumn[T any](name string, rows [][]string, column int, nulls map[string]struct{}, parse func(string) (T, error)) (dataframe.ColumnSpec, error) {
	values := make([]T, len(rows))
	validity := make([]bool, len(rows))
	nullable := false
	for row := range rows {
		text := rows[row][column]
		if _, null := nulls[text]; null {
			nullable = true
			continue
		}
		value, err := parse(text)
		if err != nil {
			return nil, fmt.Errorf("csv: row %d column %q: %w", row, name, err)
		}
		values[row] = value
		validity[row] = true
	}
	if !nullable {
		return dataframe.Column(name, values), nil
	}
	result, err := series.NewNullable(values, validity)
	if err != nil {
		panic(err)
	}
	return dataframe.ColumnFromSeries(name, result), nil
}

func makeStringSet(values []string) map[string]struct{} {
	set := make(map[string]struct{}, len(values))
	for _, value := range values {
		set[value] = struct{}{}
	}
	return set
}

func parseBool(text string) (bool, bool) {
	switch {
	case strings.EqualFold(text, "true"):
		return true, true
	case strings.EqualFold(text, "false"):
		return false, true
	default:
		return false, false
	}
}

func unmarshalValue(destination reflect.Value, text string) error {
	textUnmarshaler := reflect.TypeFor[encoding.TextUnmarshaler]()
	if destination.CanAddr() && destination.Addr().Type().Implements(textUnmarshaler) {
		return destination.Addr().Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(text))
	}
	if destination.Type().Implements(textUnmarshaler) {
		if destination.Kind() == reflect.Interface && destination.IsNil() {
			return fmt.Errorf("%w: cannot parse %v without a concrete value", dataframe.ErrUnsupported, destination.Type())
		}
		if destination.Kind() == reflect.Pointer && destination.IsNil() {
			destination.Set(reflect.New(destination.Type().Elem()))
		}
		return destination.Interface().(encoding.TextUnmarshaler).UnmarshalText([]byte(text))
	}

	switch destination.Kind() {
	case reflect.String:
		destination.SetString(text)
	case reflect.Bool:
		value, ok := parseBool(text)
		if !ok {
			return fmt.Errorf("invalid boolean %q", text)
		}
		destination.SetBool(value)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		value, err := strconv.ParseInt(text, 10, destination.Type().Bits())
		if err != nil {
			return err
		}
		destination.SetInt(value)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		value, err := strconv.ParseUint(text, 10, destination.Type().Bits())
		if err != nil {
			return err
		}
		destination.SetUint(value)
	case reflect.Float32, reflect.Float64:
		value, err := strconv.ParseFloat(text, destination.Type().Bits())
		if err != nil {
			return err
		}
		destination.SetFloat(value)
	default:
		return fmt.Errorf("%w: cannot parse %v", dataframe.ErrUnsupported, destination.Type())
	}
	return nil
}

func marshalValue(value any) (string, error) {
	return marshalReflectValue(reflect.ValueOf(value))
}

func marshalReflectValue(value reflect.Value) (string, error) {
	if !value.IsValid() {
		return "", fmt.Errorf("%w: cannot encode a present nil interface", dataframe.ErrUnsupported)
	}
	for value.Kind() == reflect.Interface {
		if value.IsNil() {
			return "", fmt.Errorf("%w: cannot encode a present nil interface", dataframe.ErrUnsupported)
		}
		value = value.Elem()
	}
	textMarshaler := reflect.TypeFor[encoding.TextMarshaler]()
	if value.Type().Implements(textMarshaler) {
		if value.Kind() == reflect.Pointer && value.IsNil() {
			return "", fmt.Errorf("%w: cannot encode nil %v", dataframe.ErrUnsupported, value.Type())
		}
		text, err := value.Interface().(encoding.TextMarshaler).MarshalText()
		return string(text), err
	}
	pointerType := reflect.PointerTo(value.Type())
	if pointerType.Implements(textMarshaler) {
		pointer := reflect.New(value.Type())
		pointer.Elem().Set(value)
		text, err := pointer.Interface().(encoding.TextMarshaler).MarshalText()
		return string(text), err
	}

	switch value.Kind() {
	case reflect.String:
		return value.String(), nil
	case reflect.Bool:
		return strconv.FormatBool(value.Bool()), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(value.Int(), 10), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.FormatUint(value.Uint(), 10), nil
	case reflect.Float32:
		return strconv.FormatFloat(value.Float(), 'g', -1, 32), nil
	case reflect.Float64:
		return strconv.FormatFloat(value.Float(), 'g', -1, 64), nil
	default:
		return "", fmt.Errorf("%w: cannot encode %v", dataframe.ErrUnsupported, value.Type())
	}
}

func validDelimiter(delimiter rune) bool {
	return delimiter != 0 && delimiter != '"' && delimiter != '\r' && delimiter != '\n' && delimiter != utf8.RuneError
}
