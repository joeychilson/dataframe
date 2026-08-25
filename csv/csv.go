// Package csv reads and writes dataframe Frames and Go struct records using
// encoding/csv-compatible configuration.
//
// CSV has no distinct null representation. The defaults interpret empty fields
// as null, so configure Reader.NullValues and Writer.NullString to the same
// token absent from present data when a round trip must distinguish null from
// an empty string.
package csv

import (
	"encoding"
	stdcsv "encoding/csv"
	"errors"
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
	// tokens do not participate in type inference. A present string equal to a
	// null token is also read as null because CSV does not preserve that
	// distinction.
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

// Read parses CSV into an inferred Frame. A nil Reader or underlying input
// returns an error. Non-null columns use the ladder bool,
// int64, float64, string. Boolean inference recognizes textual true and false,
// not numeric 0 and 1. An empty input produces the zero Frame. Header-only
// columns have non-null string type. Entirely-null columns have nullable string
// type. Ragged records, duplicate or empty headers, and values incompatible
// with a sampled type return errors.
func (r *Reader) Read() (dataframe.Frame, error) {
	if r != nil && r.InferRows < 0 {
		return dataframe.Frame{}, errors.New("csv: infer rows must not be negative")
	}
	input, err := r.readInput()
	if err != nil {
		return dataframe.Frame{}, err
	}
	if len(input.names) == 0 {
		return dataframe.Frame{}, nil
	}
	nulls := make(map[string]struct{}, len(r.NullValues))
	for _, value := range r.NullValues {
		nulls[value] = struct{}{}
	}
	sampleRows := input.rowCount
	if r.InferRows > 0 {
		sampleRows = min(sampleRows, r.InferRows)
	}
	columns := make([]dataframe.ColumnSpec, len(input.names))
	for columnIndex := range input.names {
		kind := inferString
		seen := false
		nullable := false
		nullRowsChecked := 0
		boolPossible := true
		intPossible := true
		floatPossible := true
		for row := range sampleRows {
			nullRowsChecked = row + 1
			text := input.field(row, columnIndex)
			if _, null := nulls[text]; null {
				nullable = true
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
			// A valid integer is also a valid float, so only test the wider
			// representation after integer inference fails.
			if !intPossible && floatPossible {
				_, floatErr := strconv.ParseFloat(text, 64)
				floatPossible = floatErr == nil
			}
			if !boolPossible && !intPossible && !floatPossible {
				break
			}
		}
		// Inference may stop as soon as a column is known to be text. Finish
		// checking only the unvisited rows so construction can select the final
		// nullable or non-null Series representation without another copy.
		if !nullable {
			for row := nullRowsChecked; row < input.rowCount; row++ {
				if _, null := nulls[input.field(row, columnIndex)]; null {
					nullable = true
					break
				}
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
			columns[columnIndex], err = parsedColumn(input, columnIndex, nulls, nullable, func(text string) (bool, error) {
				value, ok := parseBool(text)
				if !ok {
					return false, fmt.Errorf("invalid boolean %q", text)
				}
				return value, nil
			})
		case inferInt:
			columns[columnIndex], err = parsedColumn(input, columnIndex, nulls, nullable, func(text string) (int64, error) {
				return strconv.ParseInt(text, 10, 64)
			})
		case inferFloat:
			columns[columnIndex], err = parsedColumn(input, columnIndex, nulls, nullable, func(text string) (float64, error) {
				return strconv.ParseFloat(text, 64)
			})
		case inferString:
			columns[columnIndex], err = parsedColumn(input, columnIndex, nulls, nullable, func(text string) (string, error) {
				return text, nil
			})
		default:
			panic("csv: invalid inferred kind")
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
// their parsing. A nil Reader or underlying input returns an error.
func (r *Reader) ReadRecords[T any]() ([]T, error) {
	typeOf := reflect.TypeFor[T]()
	fields, err := record.Describe(typeOf)
	if err != nil {
		return nil, err
	}
	if r == nil || r.input == nil {
		return nil, errors.New("csv: nil reader")
	}
	reader := stdcsv.NewReader(r.input)
	reader.Comma = r.Comma
	reader.Comment = r.Comment
	reader.FieldsPerRecord = r.FieldsPerRecord
	reader.LazyQuotes = r.LazyQuotes
	reader.TrimLeadingSpace = r.TrimLeadingSpace
	reader.ReuseRecord = true

	first, err := reader.Read()
	if errors.Is(err, io.EOF) {
		return []T{}, nil
	}
	if err != nil {
		return nil, err
	}

	width := len(first)
	_, indexes, err := prepareColumnNames(first, r.Header)
	if err != nil {
		return nil, err
	}
	fieldColumns := make([]int, len(fields))
	for i, field := range fields {
		index, found := indexes[field.Name]
		if !found {
			return nil, fmt.Errorf("%w: %q", dataframe.ErrColumnNotFound, field.Name)
		}
		fieldColumns[i] = index
	}

	nulls := make(map[string]struct{}, len(r.NullValues))
	for _, value := range r.NullValues {
		nulls[value] = struct{}{}
	}
	blockRows := max(1, int(uintptr(csvRecordTargetBlockBytes)/max(uintptr(1), typeOf.Size())))
	var blocks [][]T
	var zero T
	row := 0
	useFirst := !r.Header
	for {
		var values []string
		if useFirst {
			values = first
			useFirst = false
		} else {
			values, err = reader.Read()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return nil, err
			}
		}
		if len(values) != width {
			return nil, fmt.Errorf("%w: csv row %d has %d fields, want %d", dataframe.ErrRowCount, row, len(values), width)
		}

		if row%blockRows == 0 {
			blocks = append(blocks, make([]T, 0, blockRows))
		}
		block := len(blocks) - 1
		blocks[block] = append(blocks[block], zero)
		recordValue := reflect.ValueOf(&blocks[block][len(blocks[block])-1]).Elem()
		for i, field := range fields {
			text := values[fieldColumns[i]]
			_, null := nulls[text]
			if null {
				if !field.Nullable() {
					return nil, fmt.Errorf("%w: null in non-null field %s at row %d", dataframe.ErrInvalidRecord, field.Name, row)
				}
				continue
			}
			destination := field.Destination(recordValue)
			if unmarshalErr := unmarshalValue(destination, text); unmarshalErr != nil {
				return nil, fmt.Errorf("csv: row %d column %q: %w", row, field.Name, unmarshalErr)
			}
		}
		row++
	}
	if row == 0 {
		return []T{}, nil
	}
	if len(blocks) == 1 {
		return blocks[0], nil
	}
	records := make([]T, row)
	offset := 0
	for _, block := range blocks {
		offset += copy(records[offset:], block)
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
	// empty string. Use a token absent from present encoded values when a round
	// trip must distinguish null from an empty string.
	NullString string

	output io.Writer
}

// NewWriter returns a Writer using the package defaults.
func NewWriter(w io.Writer) *Writer {
	return &Writer{Comma: ',', Header: true, output: w}
}

// Write serializes f to CSV. Values implementing encoding.TextAppender or
// encoding.TextMarshaler control their text form, with TextAppender preferred.
// Unsupported element types return an error wrapping dataframe.ErrUnsupported.
// A nil Writer or underlying output returns an error.
func (w *Writer) Write(f dataframe.Frame) error {
	writer, err := w.configuredWriter()
	if err != nil {
		return err
	}
	if f.Width() == 0 && f.Len() > 0 {
		return fmt.Errorf("%w: cannot encode %d rows without columns", dataframe.ErrUnsupported, f.Len())
	}
	if w.Header {
		if writeErr := writer.Write(f.Names()); writeErr != nil {
			return writeErr
		}
	}
	columns := slices.Collect(f.Columns())
	record := make([]string, len(columns))
	for row := range f.Len() {
		for columnIndex, column := range columns {
			value, present := column.At(row)
			if !present {
				record[columnIndex] = w.NullString
				continue
			}
			text, marshalErr := marshalReflectValue(reflect.ValueOf(value))
			if marshalErr != nil {
				return fmt.Errorf("csv: row %d column %q: %w", row, column.Name(), marshalErr)
			}
			record[columnIndex] = text
		}
		if writeErr := w.writeRecord(writer, record); writeErr != nil {
			return writeErr
		}
	}
	writer.Flush()
	return writer.Error()
}

// WriteRecords writes records of non-pointer struct type T using the same `df`
// tags as dataframe.FromRecords. Pointer and absent series.Optional fields
// write NullString. Values implementing encoding.TextAppender or
// encoding.TextMarshaler control their text form, with TextAppender preferred.
// A nil Writer or underlying output returns an error.
func (w *Writer) WriteRecords[T any](records []T) error {
	fields, err := record.Describe(reflect.TypeFor[T]())
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
		if writeErr := writer.Write(header); writeErr != nil {
			return writeErr
		}
	}

	textAppender := reflect.TypeFor[encoding.TextAppender]()
	textMarshaler := reflect.TypeFor[encoding.TextMarshaler]()
	var encoders []recordFieldEncoder
	for i, field := range fields {
		valueType := field.ValueType
		appendsText := false
		switch {
		case valueType.Implements(textAppender) || reflect.PointerTo(valueType).Implements(textAppender):
			appendsText = true
		case valueType.Implements(textMarshaler) || reflect.PointerTo(valueType).Implements(textMarshaler):
			continue
		default:
			switch valueType.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
				reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
				reflect.Float32, reflect.Float64:
			default:
				continue
			}
		}
		if encoders == nil {
			encoders = make([]recordFieldEncoder, len(fields))
		}
		encoders[i] = recordFieldEncoder{buffered: true, appendsText: appendsText}
	}

	values := reflect.ValueOf(records)
	encoded := make([]string, len(fields))
	var buffer []byte
	for row := range records {
		buffer, err = encodeRecordRow(row, values.Index(row), fields, encoders, encoded, buffer, w.NullString)
		if err != nil {
			return err
		}
		if writeErr := w.writeRecord(writer, encoded); writeErr != nil {
			return writeErr
		}
	}
	writer.Flush()
	return writer.Error()
}

// Read parses CSV using NewReader's defaults. A nil input returns an error.
func Read(r io.Reader) (dataframe.Frame, error) {
	return NewReader(r).Read()
}

// Write serializes f using NewWriter's defaults. A nil output returns an error.
func Write(w io.Writer, f dataframe.Frame) error {
	return NewWriter(w).Write(f)
}

// encoding/csv writes a one-field empty record as a blank line, which its
// Reader ignores. Quote that field explicitly so the record remains observable.
func (w *Writer) writeRecord(writer *stdcsv.Writer, record []string) error {
	if len(record) != 1 || record[0] != "" {
		return writer.Write(record)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return err
	}
	line := "\"\"\n"
	if w.UseCRLF {
		line = "\"\"\r\n"
	}
	_, err := io.WriteString(w.output, line)
	return err
}

type recordFieldEncoder struct {
	start       int
	end         int
	buffered    bool
	appendsText bool
}

func encodeRecordRow(row int, value reflect.Value, fields []record.Field, encoders []recordFieldEncoder, encoded []string, buffer []byte, nullString string) ([]byte, error) {
	buffer = buffer[:0]
	for fieldIndex, field := range fields {
		fieldValue, present := field.Extract(value)
		if !present {
			encoded[fieldIndex] = nullString
			if encoders != nil {
				encoders[fieldIndex].start = -1
			}
			continue
		}
		if encoders != nil {
			encoder := &encoders[fieldIndex]
			if encoder.buffered {
				encoder.start = len(buffer)
				var err error
				buffer, err = appendBufferedValue(buffer, fieldValue, encoder.appendsText)
				if err != nil {
					return buffer, fmt.Errorf("csv: row %d column %q: %w", row, field.Name, err)
				}
				encoder.end = len(buffer)
				continue
			}
		}
		text, err := marshalReflectValue(fieldValue)
		if err != nil {
			return buffer, fmt.Errorf("csv: row %d column %q: %w", row, field.Name, err)
		}
		encoded[fieldIndex] = text
	}
	if encoders == nil {
		return buffer, nil
	}
	text := string(buffer)
	for i, encoder := range encoders {
		if encoder.buffered && encoder.start >= 0 {
			encoded[i] = text[encoder.start:encoder.end]
		}
	}
	return buffer, nil
}

type inferredKind uint8

const (
	// Limit rows per block by record width to avoid overallocating wide inputs.
	csvInputTargetBlockFields = 4096
	// Bound temporary decoded-record blocks by physical record size. Large ignored
	// fields still contribute to T's storage and must not create oversized blocks.
	csvRecordTargetBlockBytes = 64 << 10
)

const (
	inferString inferredKind = iota
	inferBool
	inferInt
	inferFloat
)

type csvInput struct {
	names     []string
	blocks    [][]string
	rowCount  int
	blockRows int
}

func (in csvInput) field(row, column int) string {
	block := in.blocks[row/in.blockRows]
	return block[(row%in.blockRows)*len(in.names)+column]
}

func (r *Reader) readInput() (csvInput, error) {
	if r == nil || r.input == nil {
		return csvInput{}, errors.New("csv: nil reader")
	}
	reader := stdcsv.NewReader(r.input)
	reader.Comma = r.Comma
	reader.Comment = r.Comment
	reader.FieldsPerRecord = r.FieldsPerRecord
	reader.LazyQuotes = r.LazyQuotes
	reader.TrimLeadingSpace = r.TrimLeadingSpace
	reader.ReuseRecord = true

	first, err := reader.Read()
	if errors.Is(err, io.EOF) {
		return csvInput{}, nil
	}
	if err != nil {
		return csvInput{}, err
	}

	width := len(first)
	names, _, err := prepareColumnNames(first, r.Header)
	if err != nil {
		return csvInput{}, err
	}
	result := csvInput{names: names, blockRows: max(1, csvInputTargetBlockFields/width)}
	if !r.Header {
		result.blocks = append(result.blocks, make([]string, 0, result.blockRows*width))
		result.blocks[0] = append(result.blocks[0], first...)
		result.rowCount = 1
	}

	for {
		fields, readErr := reader.Read()
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return csvInput{}, readErr
		}
		if len(fields) != width {
			return csvInput{}, fmt.Errorf("%w: csv row %d has %d fields, want %d", dataframe.ErrRowCount, result.rowCount, len(fields), width)
		}
		if result.rowCount%result.blockRows == 0 {
			result.blocks = append(result.blocks, make([]string, 0, result.blockRows*width))
		}
		block := len(result.blocks) - 1
		result.blocks[block] = append(result.blocks[block], fields...)
		result.rowCount++
	}
	return result, nil
}

func prepareColumnNames(first []string, header bool) ([]string, map[string]int, error) {
	names := slices.Clone(first)
	if !header {
		for i := range names {
			names[i] = "column" + strconv.Itoa(i+1)
		}
	}
	indexes := make(map[string]int, len(names))
	for i, name := range names {
		if name == "" {
			return nil, nil, fmt.Errorf("%w: header column %d", dataframe.ErrInvalidName, i)
		}
		if _, exists := indexes[name]; exists {
			return nil, nil, fmt.Errorf("%w: header %q", dataframe.ErrColumnConflict, name)
		}
		indexes[name] = i
	}
	return names, indexes, nil
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

func parsedColumn[T any](input csvInput, column int, nulls map[string]struct{}, nullable bool, parse func(string) (T, error)) (dataframe.ColumnSpec, error) {
	name := input.names[column]
	var parseErr error
	var zero T
	if !nullable {
		values := series.NewFunc(input.rowCount, func(row int) T {
			if parseErr != nil {
				return zero
			}
			value, err := parse(input.field(row, column))
			if err != nil {
				parseErr = fmt.Errorf("csv: row %d column %q: %w", row, name, err)
				return zero
			}
			return value
		})
		if parseErr != nil {
			return nil, parseErr
		}
		return dataframe.ColumnFromSeries(name, values), nil
	}

	values := series.NewNullableFunc(input.rowCount, func(row int) (T, bool) {
		if parseErr != nil {
			return zero, false
		}
		text := input.field(row, column)
		if _, null := nulls[text]; null {
			return zero, false
		}
		value, err := parse(text)
		if err != nil {
			parseErr = fmt.Errorf("csv: row %d column %q: %w", row, name, err)
			return zero, false
		}
		return value, true
	})
	if parseErr != nil {
		return nil, parseErr
	}
	return dataframe.ColumnFromSeries(name, values), nil
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

func (w *Writer) configuredWriter() (*stdcsv.Writer, error) {
	if w == nil || w.output == nil {
		return nil, errors.New("csv: nil writer")
	}
	if !validDelimiter(w.Comma) {
		return nil, errors.New("csv: invalid field or comment delimiter")
	}
	writer := stdcsv.NewWriter(w.output)
	writer.Comma = w.Comma
	writer.UseCRLF = w.UseCRLF
	return writer, nil
}

func validDelimiter(delimiter rune) bool {
	return utf8.ValidRune(delimiter) && delimiter != 0 && delimiter != '"' && delimiter != '\r' && delimiter != '\n' && delimiter != utf8.RuneError
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
	textAppender := reflect.TypeFor[encoding.TextAppender]()
	pointerType := reflect.PointerTo(value.Type())
	if value.Type().Implements(textAppender) || pointerType.Implements(textAppender) {
		text, err := appendBufferedValue(nil, value, true)
		return string(text), err
	}
	textMarshaler := reflect.TypeFor[encoding.TextMarshaler]()
	if value.Type().Implements(textMarshaler) {
		if value.Kind() == reflect.Pointer && value.IsNil() {
			return "", fmt.Errorf("%w: cannot encode nil %v", dataframe.ErrUnsupported, value.Type())
		}
		text, err := value.Interface().(encoding.TextMarshaler).MarshalText()
		return string(text), err
	}
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

func appendBufferedValue(buffer []byte, value reflect.Value, appendsText bool) ([]byte, error) {
	if appendsText {
		for value.Kind() == reflect.Interface {
			if value.IsNil() {
				return buffer, fmt.Errorf("%w: cannot encode a present nil interface", dataframe.ErrUnsupported)
			}
			value = value.Elem()
		}
		if value.Kind() == reflect.Pointer && value.IsNil() {
			return buffer, fmt.Errorf("%w: cannot encode nil %v", dataframe.ErrUnsupported, value.Type())
		}
		textAppender := reflect.TypeFor[encoding.TextAppender]()
		if value.Type().Implements(textAppender) {
			return value.Interface().(encoding.TextAppender).AppendText(buffer)
		}
		pointerType := reflect.PointerTo(value.Type())
		if pointerType.Implements(textAppender) {
			pointer := reflect.New(value.Type())
			pointer.Elem().Set(value)
			return pointer.Interface().(encoding.TextAppender).AppendText(buffer)
		}
		return buffer, fmt.Errorf("%w: cannot encode %v", dataframe.ErrUnsupported, value.Type())
	}
	switch value.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.AppendInt(buffer, value.Int(), 10), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.AppendUint(buffer, value.Uint(), 10), nil
	case reflect.Float32:
		return strconv.AppendFloat(buffer, value.Float(), 'g', -1, 32), nil
	case reflect.Float64:
		return strconv.AppendFloat(buffer, value.Float(), 'g', -1, 64), nil
	default:
		return buffer, fmt.Errorf("%w: cannot encode %v", dataframe.ErrUnsupported, value.Type())
	}
}
