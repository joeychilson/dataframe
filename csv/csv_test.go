package csv

import (
	"encoding"
	stdcsv "encoding/csv"
	"errors"
	"fmt"
	"io"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/joeychilson/dataframe"
	recordmeta "github.com/joeychilson/dataframe/internal/record"
	"github.com/joeychilson/dataframe/series"
)

func TestReadInfersColumnsAndNulls(t *testing.T) {
	frame, err := Read(strings.NewReader("active,count,ratio,name\ntrue,1,1.5,a\nFALSE,,2,b\n"))
	if err != nil {
		t.Fatal(err)
	}
	wantSchema := []dataframe.Field{
		{Name: "active", Type: reflect.TypeFor[bool]()},
		{Name: "count", Type: reflect.TypeFor[int64](), Nullable: true},
		{Name: "ratio", Type: reflect.TypeFor[float64]()},
		{Name: "name", Type: reflect.TypeFor[string]()},
	}
	if got := frame.Schema(); !reflect.DeepEqual(got, wantSchema) {
		t.Fatalf("schema = %#v, want %#v", got, wantSchema)
	}
	active, _ := frame.Column[bool]("active")
	count, _ := frame.Column[int64]("count")
	ratio, _ := frame.Column[float64]("ratio")
	if !slices.Equal(active.Values(), []bool{true, false}) || !reflect.DeepEqual(count.Optionals(), []series.Optional[int64]{series.Some[int64](1), series.None[int64]()}) || !slices.Equal(ratio.Values(), []float64{1.5, 2}) {
		t.Fatalf("values = %v, %#v, %v", active.Values(), count.Optionals(), ratio.Values())
	}
}

func TestReadInferenceEdges_SelectNarrowestCompatibleTypes(t *testing.T) {
	mixedNumeric, err := Read(strings.NewReader("value\n1\n2.5\n"))
	if err != nil {
		t.Fatal(err)
	}
	if field := mixedNumeric.Schema()[0]; field.Type != reflect.TypeFor[float64]() || field.Nullable {
		t.Fatalf("mixed numeric field = %#v", field)
	}
	mixedValues, err := mixedNumeric.Column[float64]("value")
	if err != nil {
		t.Fatal(err)
	}
	if got := mixedValues.Values(); !slices.Equal(got, []float64{1, 2.5}) {
		t.Fatalf("mixed numeric values = %v", got)
	}

	headerOnly, err := Read(strings.NewReader("name,count\n"))
	if err != nil {
		t.Fatal(err)
	}
	if got := headerOnly.Schema(); !reflect.DeepEqual(got, []dataframe.Field{
		{Name: "name", Type: reflect.TypeFor[string]()},
		{Name: "count", Type: reflect.TypeFor[string]()},
	}) {
		t.Fatalf("header-only schema = %#v", got)
	}

	allNull, err := Read(strings.NewReader("value\n\n\"\"\n"))
	if err != nil {
		t.Fatal(err)
	}
	if field := allNull.Schema()[0]; field.Type != reflect.TypeFor[string]() || !field.Nullable {
		t.Fatalf("all-null field = %#v", field)
	}

	stringWithLateNull, err := Read(strings.NewReader("value\ntext\n\"\"\n"))
	if err != nil {
		t.Fatal(err)
	}
	if field := stringWithLateNull.Schema()[0]; field.Type != reflect.TypeFor[string]() || !field.Nullable {
		t.Fatalf("string with late null field = %#v", field)
	}

	reader := NewReader(strings.NewReader("1,a\n0,b\n"))
	reader.Header = false
	frame, err := reader.Read()
	if err != nil {
		t.Fatal(err)
	}
	if got := frame.Names(); !slices.Equal(got, []string{"column1", "column2"}) {
		t.Fatalf("synthesized names = %v", got)
	}
	if frame.Schema()[0].Type != reflect.TypeFor[int64]() {
		t.Fatalf("numeric 0/1 inferred as %v", frame.Schema()[0].Type)
	}
	values, err := frame.Column[int64]("column1")
	if err != nil {
		t.Fatal(err)
	}
	if got := values.Values(); !slices.Equal(got, []int64{1, 0}) {
		t.Fatalf("headerless values = %v", got)
	}

	reader = NewReader(strings.NewReader("value\n\"\"\n"))
	reader.NullValues = nil
	frame, err = reader.Read()
	if err != nil {
		t.Fatal(err)
	}
	if frame.Schema()[0].Nullable || frame.Schema()[0].Type != reflect.TypeFor[string]() {
		t.Fatalf("disabled null recognition schema = %#v", frame.Schema()[0])
	}
}

func TestReaderConfiguration_MatchesEncodingCSV(t *testing.T) {
	const input = "# ignored\nid; value\n1; a\"b\n"
	newReader := func() *Reader {
		reader := NewReader(strings.NewReader(input))
		reader.Comma = ';'
		reader.Comment = '#'
		reader.LazyQuotes = true
		reader.TrimLeadingSpace = true
		return reader
	}

	frame, err := newReader().Read()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := frame.Names(), []string{"id", "value"}; !slices.Equal(got, want) {
		t.Fatalf("frame names = %v, want %v", got, want)
	}
	values, err := frame.Column[string]("value")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := values.Values(), []string{"a\"b"}; !slices.Equal(got, want) {
		t.Fatalf("frame values = %v, want %v", got, want)
	}

	type row struct {
		ID    int    `df:"id"`
		Value string `df:"value"`
	}
	records, err := newReader().ReadRecords[row]()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := records, []row{{ID: 1, Value: "a\"b"}}; !slices.Equal(got, want) {
		t.Fatalf("records = %v, want %v", got, want)
	}
}

func TestRead_RespectsInferenceLimitAndReportsInvalidInput(t *testing.T) {
	reader := NewReader(strings.NewReader("value\n1\n\"\"\n2\n"))
	reader.InferRows = 1
	frame, err := reader.Read()
	if err != nil {
		t.Fatal(err)
	}
	values, err := frame.Column[int64]("value")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := values.Optionals(), []series.Optional[int64]{series.Some[int64](1), series.None[int64](), series.Some[int64](2)}; !reflect.DeepEqual(got, want) {
		t.Fatalf("sampled nullable values = %#v, want %#v", got, want)
	}

	reader = NewReader(strings.NewReader("value\n1\nbad\n"))
	reader.InferRows = 1
	if _, readErr := reader.Read(); readErr == nil || !strings.Contains(readErr.Error(), "row 1") {
		t.Fatalf("sampled integer error = %v", readErr)
	}

	reader = NewReader(strings.NewReader("value\n1\n"))
	reader.InferRows = -1
	if _, readErr := reader.Read(); readErr == nil || readErr.Error() != "csv: infer rows must not be negative" {
		t.Fatalf("negative InferRows error = %v", readErr)
	}

	reader = NewReader(strings.NewReader("a,b\n1\n"))
	reader.FieldsPerRecord = -1
	if _, readErr := reader.Read(); !errors.Is(readErr, dataframe.ErrRowCount) {
		t.Fatalf("ragged error = %v", readErr)
	}
	if _, readErr := Read(strings.NewReader("a,a\n1,2\n")); !errors.Is(readErr, dataframe.ErrColumnConflict) {
		t.Fatalf("duplicate header error = %v", readErr)
	}
	if _, readErr := Read(strings.NewReader("\"\",b\n1,2\n")); !errors.Is(readErr, dataframe.ErrInvalidName) {
		t.Fatalf("empty header error = %v", readErr)
	}
	for _, test := range []struct {
		name  string
		input string
		want  error
	}{
		{name: "reports header conflict before a ragged row", input: "a,a\n1\n", want: dataframe.ErrColumnConflict},
		{name: "reports header conflict before a syntax error", input: "a,a\n\"unterminated\n", want: dataframe.ErrColumnConflict},
		{name: "reports row count before a later syntax error", input: "a,b\n1\n\"unterminated\n", want: dataframe.ErrRowCount},
	} {
		t.Run(test.name, func(t *testing.T) {
			testReader := NewReader(strings.NewReader(test.input))
			testReader.FieldsPerRecord = -1
			if _, readErr := testReader.Read(); !errors.Is(readErr, test.want) {
				t.Fatalf("Read() error = %v, want %v", readErr, test.want)
			}
		})
	}
}

func TestReadAcrossInputBlocks(t *testing.T) {
	blockRows := csvInputTargetBlockFields / 2
	var input strings.Builder
	input.WriteString("id,name\n")
	for i := range blockRows + 1 {
		fmt.Fprintf(&input, "%d,value-%d\n", i, i)
	}
	frame, err := Read(strings.NewReader(input.String()))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := frame.Len(), blockRows+1; got != want {
		t.Fatalf("row count = %d, want %d", got, want)
	}
	ids, err := frame.Column[int64]("id")
	if err != nil {
		t.Fatal(err)
	}
	names, err := frame.Column[string]("name")
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []int{blockRows - 1, blockRows} {
		if value, ok := ids.At(row); !ok || value != int64(row) {
			t.Fatalf("id at row %d = %d, %t", row, value, ok)
		}
		if value, ok := names.At(row); !ok || value != fmt.Sprintf("value-%d", row) {
			t.Fatalf("name at row %d = %q, %t", row, value, ok)
		}
	}
}

func TestReadRecordsAcrossBlocks(t *testing.T) {
	type row struct {
		ID      int        `df:"id"`
		Padding [1024]byte `df:"-"`
	}
	blockRows := max(1, int(uintptr(csvRecordTargetBlockBytes)/reflect.TypeFor[row]().Size()))
	var input strings.Builder
	input.WriteString("id\n")
	for i := range blockRows + 1 {
		fmt.Fprintln(&input, i)
	}
	records, err := NewReader(strings.NewReader(input.String())).ReadRecords[row]()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(records), blockRows+1; got != want {
		t.Fatalf("record count = %d, want %d", got, want)
	}
	for _, index := range []int{blockRows - 1, blockRows} {
		if records[index].ID != index {
			t.Fatalf("record %d ID = %d", index, records[index].ID)
		}
	}
}

func TestReadRecords_UsesRecordSchemaAndTextDecoders(t *testing.T) {
	type row struct {
		ID     int                      `df:"id"`
		Code   textCode                 `df:"code"`
		Name   *string                  `df:"name"`
		Score  series.Optional[float64] `df:"score"`
		Active bool                     `df:"active"`
	}
	records, err := NewReader(strings.NewReader("id,code,name,score,active,ignored\n1,C7,A,1.5,true,x\n2,C8,,,false,y\n")).ReadRecords[row]()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 || records[0].Code != 7 || records[0].Name == nil || *records[0].Name != "A" || records[1].Name != nil || !records[0].Score.Valid || records[1].Score.Valid || records[1].Active {
		t.Fatalf("records = %#v", records)
	}
	empty, err := NewReader(strings.NewReader("")).ReadRecords[row]()
	if err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("empty records = %#v, error %v", empty, err)
	}

	headerlessReader := NewReader(strings.NewReader("1,a\n2,b\n"))
	headerlessReader.Header = false
	headerless, err := headerlessReader.ReadRecords[struct {
		ID   int    `df:"column1"`
		Name string `df:"column2"`
	}]()
	if err != nil {
		t.Fatal(err)
	}
	if len(headerless) != 2 || headerless[0].ID != 1 || headerless[1].Name != "b" {
		t.Fatalf("headerless records = %#v", headerless)
	}

	raggedReader := NewReader(strings.NewReader("id,name\n1,a\n2\n"))
	raggedReader.FieldsPerRecord = -1
	if _, readErr := raggedReader.ReadRecords[struct {
		ID   int    `df:"id"`
		Name string `df:"name"`
	}](); !errors.Is(readErr, dataframe.ErrRowCount) {
		t.Fatalf("ragged record error = %v", readErr)
	}

	if _, readErr := NewReader(strings.NewReader("id\n\"\"\n")).ReadRecords[struct {
		ID int `df:"id"`
	}](); !errors.Is(readErr, dataframe.ErrInvalidRecord) {
		t.Fatalf("null scalar error = %v", readErr)
	}
	if _, readErr := NewReader(strings.NewReader("other\n1\n")).ReadRecords[struct {
		ID int `df:"id"`
	}](); !errors.Is(readErr, dataframe.ErrColumnNotFound) {
		t.Fatalf("missing field error = %v", readErr)
	}
	if _, readErr := NewReader(strings.NewReader("value\nx\n")).ReadRecords[struct {
		Value complex64 `df:"value"`
	}](); !errors.Is(readErr, dataframe.ErrUnsupported) {
		t.Fatalf("unsupported field error = %v", readErr)
	}
	if _, readErr := NewReader(strings.NewReader("value\nx\n")).ReadRecords[struct {
		Value encoding.TextUnmarshaler `df:"value"`
	}](); !errors.Is(readErr, dataframe.ErrUnsupported) {
		t.Fatalf("nil TextUnmarshaler interface error = %v", readErr)
	}
}

func TestWrite_EncodesFramesAndTextValues(t *testing.T) {
	names := series.FromOptionals([]series.Optional[string]{series.Some("a,b"), series.None[string]()})
	frame, err := dataframe.New(
		dataframe.Column("id", []int{1, 2}),
		dataframe.ColumnFromSeries("name", names),
		dataframe.Column("active", []bool{true, false}),
	)
	if err != nil {
		t.Fatal(err)
	}
	var output strings.Builder
	writer := NewWriter(&output)
	writer.NullString = "NULL"
	if writeErr := writer.Write(frame); writeErr != nil {
		t.Fatal(writeErr)
	}
	want := "id,name,active\n1,\"a,b\",true\n2,NULL,false\n"
	if got := output.String(); got != want {
		t.Fatalf("CSV = %q, want %q", got, want)
	}

	unsupported, err := dataframe.New(dataframe.Column("value", [][]int{{1}}))
	if err != nil {
		t.Fatal(err)
	}
	if writeErr := Write(&strings.Builder{}, unsupported); !errors.Is(writeErr, dataframe.ErrUnsupported) {
		t.Fatalf("unsupported value error = %v", writeErr)
	}
}

func TestWrite_PreservesSingleEmptyFieldRows(t *testing.T) {
	values := series.FromOptionals([]series.Optional[string]{series.Some(""), series.None[string]()})
	frame, err := dataframe.New(dataframe.ColumnFromSeries("value", values))
	if err != nil {
		t.Fatal(err)
	}

	var output strings.Builder
	writer := NewWriter(&output)
	writer.NullString = "NULL"
	if writeErr := writer.Write(frame); writeErr != nil {
		t.Fatal(writeErr)
	}
	if got, want := output.String(), "value\n\"\"\nNULL\n"; got != want {
		t.Fatalf("CSV = %q, want %q", got, want)
	}

	reader := NewReader(strings.NewReader(output.String()))
	reader.NullValues = []string{"NULL"}
	roundTrip, err := reader.Read()
	if err != nil {
		t.Fatal(err)
	}
	got, err := roundTrip.Column[string]("value")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Optionals(), values.Optionals()) {
		t.Fatalf("round-trip values = %#v, want %#v", got.Optionals(), values.Optionals())
	}
}

func TestWriteRecords_UsesTextEncoders(t *testing.T) {
	type row struct {
		Code      textCode             `df:"code"`
		Preferred preferredTextEncoder `df:"preferred"`
		Label     appendOnlyText       `df:"label"`
		Name      *string              `df:"name"`
	}
	name := "A"
	appendCalls := 0
	preferred := preferredTextEncoder{AppendCalls: &appendCalls}
	var output strings.Builder
	if err := NewWriter(&output).WriteRecords([]row{{Code: 7, Preferred: preferred, Label: "first", Name: &name}, {Code: 8, Preferred: preferred, Label: ""}}); err != nil {
		t.Fatal(err)
	}
	if got, want := output.String(), "code,preferred,label,name\nC7,P,first,A\nC8,P,,\n"; got != want {
		t.Fatalf("CSV = %q, want %q", got, want)
	}
	if appendCalls != 2 {
		t.Fatalf("AppendText calls = %d, want 2", appendCalls)
	}

	frame, err := dataframe.New(dataframe.Column("preferred", []preferredTextEncoder{preferred}))
	if err != nil {
		t.Fatal(err)
	}
	output.Reset()
	if writeErr := NewWriter(&output).Write(frame); writeErr != nil {
		t.Fatal(writeErr)
	}
	if got, want := output.String(), "preferred\nP\n"; got != want {
		t.Fatalf("frame CSV = %q, want %q", got, want)
	}
	if appendCalls != 3 {
		t.Fatalf("frame AppendText calls = %d, want 3", appendCalls)
	}
}

func TestWriteRecords_PreservesSingleEmptyFieldRows(t *testing.T) {
	type row struct {
		Value *string `df:"value"`
	}
	empty := ""
	records := []row{{Value: &empty}, {}}

	var output strings.Builder
	writer := NewWriter(&output)
	writer.UseCRLF = true
	writer.NullString = "NULL"
	if err := writer.WriteRecords(records); err != nil {
		t.Fatal(err)
	}
	if got, want := output.String(), "value\r\n\"\"\r\nNULL\r\n"; got != want {
		t.Fatalf("CSV = %q, want %q", got, want)
	}

	reader := NewReader(strings.NewReader(output.String()))
	reader.NullValues = []string{"NULL"}
	got, err := reader.ReadRecords[row]()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, records) {
		t.Fatalf("round-trip records = %#v, want %#v", got, records)
	}
}

func TestWriteRecords_UsesRecordSchemaAndEncodesNulls(t *testing.T) {
	type Metadata struct {
		ID int `df:"id"`
	}
	type row struct {
		Metadata
		Name    series.Optional[string] `df:"name"`
		Ignored string                  `df:"-"`
	}
	records := []row{
		{ID: 1, Name: series.Some("A")},
		{ID: 2},
	}
	var output strings.Builder
	writer := NewWriter(&output)
	writer.Comma = ';'
	writer.UseCRLF = true
	writer.NullString = "NULL"
	if err := writer.WriteRecords(records); err != nil {
		t.Fatal(err)
	}
	if got, want := output.String(), "id;name\r\n1;A\r\n2;NULL\r\n"; got != want {
		t.Fatalf("CSV = %q, want %q", got, want)
	}

	output.Reset()
	if err := NewWriter(&output).WriteRecords([]row{}); err != nil {
		t.Fatal(err)
	}
	if got, want := output.String(), "id,name\n"; got != want {
		t.Fatalf("empty CSV = %q, want %q", got, want)
	}
	if err := (*Writer)(nil).WriteRecords([]int{}); !errors.Is(err, dataframe.ErrInvalidRecord) {
		t.Fatalf("invalid record error = %v", err)
	}
}

func TestWriteRecordsFormatsNumericFields(t *testing.T) {
	type count int16
	type row struct {
		Signed   count
		Unsigned uint32
		Ratio    *float32
		Value    float64
	}
	ratio := float32(1.25)
	records := []row{
		{Signed: -2, Unsigned: 3, Ratio: &ratio, Value: 2.5},
		{Signed: 4, Unsigned: 5, Value: 6.75},
	}
	var output strings.Builder
	writer := NewWriter(&output)
	writer.NullString = "NULL"
	if err := writer.WriteRecords(records); err != nil {
		t.Fatal(err)
	}
	if got, want := output.String(), "Signed,Unsigned,Ratio,Value\n-2,3,1.25,2.5\n4,5,NULL,6.75\n"; got != want {
		t.Fatalf("CSV = %q, want %q", got, want)
	}
}

func TestWriteRecordsDoesNotMutateTextMarshaler(t *testing.T) {
	type row struct {
		Code incrementingTextCode `df:"code"`
	}
	records := []row{{Code: 7}}
	var output strings.Builder
	writer := NewWriter(&output)
	writer.Header = false
	if err := writer.WriteRecords(records); err != nil {
		t.Fatal(err)
	}
	if got, want := output.String(), "C8\n"; got != want {
		t.Fatalf("CSV = %q, want %q", got, want)
	}
	if records[0].Code != 7 {
		t.Fatalf("record code mutated to %d", records[0].Code)
	}
}

func TestWriteRecordsDoesNotMutateTextAppender(t *testing.T) {
	type row struct {
		Code incrementingAppendCode `df:"code"`
	}
	records := []row{{Code: 7}}
	var output strings.Builder
	writer := NewWriter(&output)
	writer.Header = false
	if err := writer.WriteRecords(records); err != nil {
		t.Fatal(err)
	}
	if got, want := output.String(), "C8\n"; got != want {
		t.Fatalf("CSV = %q, want %q", got, want)
	}
	if records[0].Code != 7 {
		t.Fatalf("record code mutated to %d", records[0].Code)
	}
}

func TestWriteRecordsTreatsNilInterfacesAsPresent(t *testing.T) {
	tests := []struct {
		name  string
		write func(*Writer) error
	}{
		{
			name: "rejects a nil plain interface",
			write: func(writer *Writer) error {
				return writer.WriteRecords([]struct {
					Value any `df:"value"`
				}{{}})
			},
		},
		{
			name: "rejects a nil optional interface",
			write: func(writer *Writer) error {
				return writer.WriteRecords([]struct {
					Value series.Optional[any] `df:"value"`
				}{{Value: series.Some[any](nil)}})
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.write(NewWriter(&strings.Builder{})); !errors.Is(err, dataframe.ErrUnsupported) {
				t.Fatalf("present nil error = %v, want ErrUnsupported", err)
			}
		})
	}
}

func TestWriteRejectsNonemptyZeroWidthData(t *testing.T) {
	frame, err := dataframe.FromRecords([]struct{}{{}, {}})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name  string
		write func() error
	}{
		{name: "rejects a nonempty frame", write: func() error { return Write(&strings.Builder{}, frame) }},
		{name: "rejects nonempty records", write: func() error { return NewWriter(&strings.Builder{}).WriteRecords([]struct{}{{}, {}}) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if writeErr := test.write(); !errors.Is(writeErr, dataframe.ErrUnsupported) {
				t.Fatalf("error = %v, want ErrUnsupported", writeErr)
			}
		})
	}

	if writeErr := Write(&strings.Builder{}, dataframe.Frame{}); writeErr != nil {
		t.Fatalf("empty frame error = %v", writeErr)
	}
	if writeErr := NewWriter(&strings.Builder{}).WriteRecords([]struct{}{}); writeErr != nil {
		t.Fatalf("empty records error = %v", writeErr)
	}
}

func TestWrite_UsesDefaultsWhileWriterHonorsConfiguration(t *testing.T) {
	frame, err := dataframe.New(
		dataframe.Column("id", []int{1}),
		dataframe.Column("name", []string{"x"}),
	)
	if err != nil {
		t.Fatal(err)
	}
	var output strings.Builder
	writer := NewWriter(&output)
	writer.Comma = ';'
	writer.UseCRLF = true
	if writeErr := writer.Write(frame); writeErr != nil {
		t.Fatal(writeErr)
	}
	if output.String() != "id;name\r\n1;x\r\n" {
		t.Fatalf("configured output = %q", output.String())
	}
	output.Reset()
	if writeErr := Write(&output, frame); writeErr != nil {
		t.Fatal(writeErr)
	}
	if output.String() != "id,name\n1,x\n" {
		t.Fatalf("convenience output = %q", output.String())
	}
	if writeErr := (*Writer)(nil).Write(frame); writeErr == nil {
		t.Fatal("nil Writer did not fail")
	}
}

func TestWriterRejectsInvalidUnicodeDelimiter(t *testing.T) {
	frame, err := dataframe.New(dataframe.Column("value", []int{}))
	if err != nil {
		t.Fatal(err)
	}
	for _, delimiter := range []rune{-1, 0xD800, utf8.MaxRune + 1} {
		writer := NewWriter(&strings.Builder{})
		writer.Header = false
		writer.Comma = delimiter
		if writeErr := writer.Write(frame); writeErr == nil {
			t.Errorf("Write() accepted delimiter %U", delimiter)
		}

		recordWriter := NewWriter(&strings.Builder{})
		recordWriter.Header = false
		recordWriter.Comma = delimiter
		if writeErr := recordWriter.WriteRecords([]struct{ Value int }{}); writeErr == nil {
			t.Errorf("WriteRecords() accepted delimiter %U", delimiter)
		}
	}
}

func FuzzRecordRoundTrip(f *testing.F) {
	type record struct {
		Appended  appendOnlyText
		Both      textCode
		Marshaled marshalTextCode
		Name      *string
		Score     series.Optional[int]
	}
	f.Add([]byte{0, 1, 2, 3, 7, 15, 31, 63, 127, 255})
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, data []byte) {
		data = data[:min(len(data), 32)]
		labels := [...]string{"", "plain", "a,b", "a\"b", "line\nbreak"}
		records := make([]record, len(data))
		for i, value := range data {
			records[i].Appended = appendOnlyText(labels[int(value)%len(labels)])
			records[i].Both = textCode(int8(value))
			records[i].Marshaled = marshalTextCode(int8(value))
			if value&1 != 0 {
				name := labels[int(value>>1)%len(labels)]
				records[i].Name = &name
			}
			if value&2 != 0 {
				records[i].Score = series.Some(int(int8(value)))
			}
		}

		var output strings.Builder
		writer := NewWriter(&output)
		writer.NullString = "<NULL>"
		if err := writer.WriteRecords(records); err != nil {
			t.Fatal(err)
		}
		reader := NewReader(strings.NewReader(output.String()))
		reader.NullValues = []string{"<NULL>"}
		got, err := reader.ReadRecords[record]()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, records) {
			t.Fatalf("round trip = %#v, want %#v\nCSV:\n%s", got, records, output.String())
		}
	})
}

type textCode int

func (c *textCode) UnmarshalText(text []byte) error {
	value := strings.TrimPrefix(string(text), "C")
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return err
	}
	*c = textCode(parsed)
	return nil
}

func (c textCode) AppendText(buffer []byte) ([]byte, error) {
	return strconv.AppendInt(append(buffer, 'C'), int64(c), 10), nil
}

func (c textCode) MarshalText() ([]byte, error) {
	return c.AppendText(nil)
}

type preferredTextEncoder struct {
	AppendCalls *int
}

func (e preferredTextEncoder) AppendText(buffer []byte) ([]byte, error) {
	(*e.AppendCalls)++
	return append(buffer, 'P'), nil
}

func (preferredTextEncoder) MarshalText() ([]byte, error) {
	return []byte{'P'}, nil
}

type appendOnlyText string

func (s appendOnlyText) AppendText(buffer []byte) ([]byte, error) {
	return append(buffer, s...), nil
}

type incrementingTextCode int

func (c *incrementingTextCode) MarshalText() ([]byte, error) {
	*c = *c + 1
	return fmt.Appendf(nil, "C%d", *c), nil
}

type incrementingAppendCode int

func (c *incrementingAppendCode) AppendText(buffer []byte) ([]byte, error) {
	*c = *c + 1
	return fmt.Appendf(buffer, "C%d", *c), nil
}

type marshalTextCode int

func (c *marshalTextCode) UnmarshalText(text []byte) error {
	value := strings.TrimPrefix(string(text), "M")
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return err
	}
	*c = marshalTextCode(parsed)
	return nil
}

func (c marshalTextCode) MarshalText() ([]byte, error) {
	return strconv.AppendInt([]byte{'M'}, int64(c), 10), nil
}

func BenchmarkRead(b *testing.B) {
	var input strings.Builder
	input.WriteString("id,value\n")
	for i := range 10_000 {
		fmt.Fprintf(&input, "%d,%f\n", i, float64(i)/3)
	}
	text := input.String()
	b.SetBytes(int64(len(text)))
	b.ReportAllocs()
	var result dataframe.Frame
	for b.Loop() {
		var err error
		result, err = Read(strings.NewReader(text))
		if err != nil {
			b.Fatal(err)
		}
	}
	runtime.KeepAlive(result)
}

func BenchmarkCSVInputStorage(b *testing.B) {
	for _, input := range []struct {
		name  string
		rows  int
		width int
	}{
		{name: "Narrow", rows: 10_000, width: 2},
		{name: "Wide", rows: 1_000, width: 64},
	} {
		var text strings.Builder
		writer := stdcsv.NewWriter(&text)
		header := make([]string, input.width)
		for i := range header {
			header[i] = "column" + strconv.Itoa(i)
		}
		if writeErr := writer.Write(header); writeErr != nil {
			b.Fatal(writeErr)
		}
		record := make([]string, input.width)
		for row := range input.rows {
			for column := range record {
				record[column] = strconv.Itoa(row + column)
			}
			if writeErr := writer.Write(record); writeErr != nil {
				b.Fatal(writeErr)
			}
		}
		writer.Flush()
		if writeErr := writer.Error(); writeErr != nil {
			b.Fatal(writeErr)
		}
		encoded := text.String()
		b.Run(input.name, func(b *testing.B) {
			b.SetBytes(int64(len(encoded)))
			b.Run("Optimized", func(b *testing.B) {
				b.ReportAllocs()
				var result csvInput
				for b.Loop() {
					var readErr error
					result, readErr = NewReader(strings.NewReader(encoded)).readInput()
					if readErr != nil {
						b.Fatal(readErr)
					}
				}
				runtime.KeepAlive(result)
			})
			b.Run("Reference", func(b *testing.B) {
				b.ReportAllocs()
				var result [][]string
				for b.Loop() {
					reader := stdcsv.NewReader(strings.NewReader(encoded))
					reader.ReuseRecord = true
					if _, readErr := reader.Read(); readErr != nil {
						b.Fatal(readErr)
					}
					result = nil
					for {
						fields, readErr := reader.Read()
						if errors.Is(readErr, io.EOF) {
							break
						}
						if readErr != nil {
							b.Fatal(readErr)
						}
						result = append(result, slices.Clone(fields))
					}
				}
				runtime.KeepAlive(result)
			})
		})
	}
}

func BenchmarkReadRecords(b *testing.B) {
	type row struct {
		ID    int     `df:"id"`
		Value float64 `df:"value"`
	}
	var input strings.Builder
	input.WriteString("id,value\n")
	for i := range 10_000 {
		fmt.Fprintf(&input, "%d,%f\n", i, float64(i)/3)
	}
	text := input.String()
	b.SetBytes(int64(len(text)))
	b.ReportAllocs()
	var result []row
	for b.Loop() {
		var err error
		result, err = NewReader(strings.NewReader(text)).ReadRecords[row]()
		if err != nil {
			b.Fatal(err)
		}
	}
	runtime.KeepAlive(result)
}

func BenchmarkCSVRecordStorage(b *testing.B) {
	type smallRecord struct {
		ID    int
		Value float64
	}
	type largeRecord struct {
		ID      int
		Padding [1024]byte
	}
	b.Run("Small", func(b *testing.B) {
		const count = 10_000
		source := make([]smallRecord, count)
		blockRows := max(1, int(uintptr(csvRecordTargetBlockBytes)/reflect.TypeFor[smallRecord]().Size()))
		b.Run("Optimized", func(b *testing.B) {
			b.ReportAllocs()
			var result []smallRecord
			for b.Loop() {
				var blocks [][]smallRecord
				for row, value := range source {
					if row%blockRows == 0 {
						blocks = append(blocks, make([]smallRecord, 0, blockRows))
					}
					block := len(blocks) - 1
					blocks[block] = append(blocks[block], value)
				}
				result = make([]smallRecord, count)
				offset := 0
				for _, block := range blocks {
					offset += copy(result[offset:], block)
				}
			}
			runtime.KeepAlive(result)
		})
		b.Run("Reference", func(b *testing.B) {
			b.ReportAllocs()
			var result []smallRecord
			for b.Loop() {
				result = slices.Clone(source)
			}
			runtime.KeepAlive(result)
		})
	})
	b.Run("Large", func(b *testing.B) {
		const count = 1_000
		source := make([]largeRecord, count)
		blockRows := max(1, int(uintptr(csvRecordTargetBlockBytes)/reflect.TypeFor[largeRecord]().Size()))
		b.Run("Optimized", func(b *testing.B) {
			b.ReportAllocs()
			var result []largeRecord
			for b.Loop() {
				var blocks [][]largeRecord
				for row, value := range source {
					if row%blockRows == 0 {
						blocks = append(blocks, make([]largeRecord, 0, blockRows))
					}
					block := len(blocks) - 1
					blocks[block] = append(blocks[block], value)
				}
				result = make([]largeRecord, count)
				offset := 0
				for _, block := range blocks {
					offset += copy(result[offset:], block)
				}
			}
			runtime.KeepAlive(result)
		})
		b.Run("Reference", func(b *testing.B) {
			b.ReportAllocs()
			var result []largeRecord
			for b.Loop() {
				result = slices.Clone(source)
			}
			runtime.KeepAlive(result)
		})
	})
}

func BenchmarkWrite(b *testing.B) {
	frame, err := dataframe.New(
		dataframe.Column("id", make([]int, 10_000)),
		dataframe.Column("value", make([]float64, 10_000)),
	)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	var outputLength int
	for b.Loop() {
		var output strings.Builder
		if writeErr := Write(&output, frame); writeErr != nil {
			b.Fatal(writeErr)
		}
		outputLength = output.Len()
	}
	runtime.KeepAlive(outputLength)
}

func BenchmarkWriteRecords(b *testing.B) {
	type row struct {
		ID    int
		Value float64
	}
	records := make([]row, 10_000)
	for i := range records {
		records[i] = row{ID: i, Value: float64(i) / 3}
	}
	fields, describeErr := recordmeta.Describe(reflect.TypeFor[row]())
	if describeErr != nil {
		b.Fatal(describeErr)
	}
	implementations := []struct {
		name  string
		write func(*strings.Builder) error
	}{
		{name: "Optimized", write: func(output *strings.Builder) error {
			return NewWriter(output).WriteRecords(records)
		}},
		{name: "Reference", write: func(output *strings.Builder) error {
			writer := stdcsv.NewWriter(output)
			header := make([]string, len(fields))
			for i, field := range fields {
				header[i] = field.Name
			}
			if writeErr := writer.Write(header); writeErr != nil {
				return writeErr
			}
			values := reflect.ValueOf(records)
			encoded := make([]string, len(fields))
			for rowIndex := range records {
				for fieldIndex, field := range fields {
					value, _ := field.Extract(values.Index(rowIndex))
					text, marshalErr := marshalReflectValue(value)
					if marshalErr != nil {
						return marshalErr
					}
					encoded[fieldIndex] = text
				}
				if writeErr := writer.Write(encoded); writeErr != nil {
					return writeErr
				}
			}
			writer.Flush()
			return writer.Error()
		}},
	}
	var optimizedOutput, referenceOutput strings.Builder
	if writeErr := implementations[0].write(&optimizedOutput); writeErr != nil {
		b.Fatal(writeErr)
	}
	if writeErr := implementations[1].write(&referenceOutput); writeErr != nil {
		b.Fatal(writeErr)
	}
	if optimizedOutput.String() != referenceOutput.String() {
		b.Fatal("buffered record encoding differs from reference")
	}
	for _, implementation := range implementations {
		b.Run(implementation.name, func(b *testing.B) {
			b.ReportAllocs()
			var outputLength int
			for b.Loop() {
				var output strings.Builder
				if writeErr := implementation.write(&output); writeErr != nil {
					b.Fatal(writeErr)
				}
				outputLength = output.Len()
			}
			runtime.KeepAlive(outputLength)
		})
	}
}

func BenchmarkWriteRecordsTextAppender(b *testing.B) {
	type row struct {
		Code textCode
	}
	records := make([]row, 10_000)
	for i := range records {
		records[i].Code = textCode(i)
	}
	fields, describeErr := recordmeta.Describe(reflect.TypeFor[row]())
	if describeErr != nil {
		b.Fatal(describeErr)
	}
	implementations := []struct {
		name  string
		write func(*strings.Builder) error
	}{
		{name: "Optimized", write: func(output *strings.Builder) error {
			return NewWriter(output).WriteRecords(records)
		}},
		{name: "Reference", write: func(output *strings.Builder) error {
			writer := stdcsv.NewWriter(output)
			header := []string{fields[0].Name}
			if writeErr := writer.Write(header); writeErr != nil {
				return writeErr
			}
			values := reflect.ValueOf(records)
			encoded := make([]string, 1)
			for rowIndex := range records {
				value, _ := fields[0].Extract(values.Index(rowIndex))
				text, marshalErr := marshalReflectValue(value)
				if marshalErr != nil {
					return marshalErr
				}
				encoded[0] = text
				if writeErr := writer.Write(encoded); writeErr != nil {
					return writeErr
				}
			}
			writer.Flush()
			return writer.Error()
		}},
	}
	var optimizedOutput, referenceOutput strings.Builder
	if writeErr := implementations[0].write(&optimizedOutput); writeErr != nil {
		b.Fatal(writeErr)
	}
	if writeErr := implementations[1].write(&referenceOutput); writeErr != nil {
		b.Fatal(writeErr)
	}
	if optimizedOutput.String() != referenceOutput.String() {
		b.Fatal("buffered TextAppender encoding differs from reference")
	}
	for _, implementation := range implementations {
		b.Run(implementation.name, func(b *testing.B) {
			b.ReportAllocs()
			var outputLength int
			for b.Loop() {
				var output strings.Builder
				if writeErr := implementation.write(&output); writeErr != nil {
					b.Fatal(writeErr)
				}
				outputLength = output.Len()
			}
			runtime.KeepAlive(outputLength)
		})
	}
}
