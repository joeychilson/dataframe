package csv

import (
	"encoding"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/joeychilson/dataframe"
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

func TestReadInferenceEdges(t *testing.T) {
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

func TestReadInferRowsAndErrors(t *testing.T) {
	reader := NewReader(strings.NewReader("value\n1\nbad\n"))
	reader.InferRows = 1
	if _, err := reader.Read(); err == nil {
		t.Fatal("sampled integer accepted an incompatible later value")
	}

	reader = NewReader(strings.NewReader("value\n1\n"))
	reader.InferRows = -1
	if _, err := reader.Read(); err == nil {
		t.Fatal("negative InferRows did not fail")
	}

	reader = NewReader(strings.NewReader("a,b\n1\n"))
	reader.FieldsPerRecord = -1
	if _, err := reader.Read(); !errors.Is(err, dataframe.ErrRowCount) {
		t.Fatalf("ragged error = %v", err)
	}
	if _, err := Read(strings.NewReader("a,a\n1,2\n")); !errors.Is(err, dataframe.ErrColumnConflict) {
		t.Fatalf("duplicate header error = %v", err)
	}
	if _, err := Read(strings.NewReader("\"\",b\n1,2\n")); !errors.Is(err, dataframe.ErrInvalidName) {
		t.Fatalf("empty header error = %v", err)
	}
	reader = NewReader(strings.NewReader("a,a\n1\n"))
	reader.FieldsPerRecord = -1
	if _, err := reader.Read(); !errors.Is(err, dataframe.ErrColumnConflict) {
		t.Fatalf("header and ragged error = %v", err)
	}
}

func TestReadAcrossInputBlocks(t *testing.T) {
	var input strings.Builder
	input.WriteString("id,name\n")
	for i := 0; i <= csvInputTargetBlockFields/2; i++ {
		fmt.Fprintf(&input, "%d,value\n", i)
	}
	frame, err := Read(strings.NewReader(input.String()))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := frame.Len(), csvInputTargetBlockFields/2+1; got != want {
		t.Fatalf("row count = %d, want %d", got, want)
	}
	ids, err := frame.Column[int64]("id")
	if err != nil {
		t.Fatal(err)
	}
	if value, _ := ids.At(ids.Len() - 1); value != csvInputTargetBlockFields/2 {
		t.Fatalf("last id = %d, want %d", value, csvInputTargetBlockFields/2)
	}
}

func TestReadRecords(t *testing.T) {
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

	if _, err := NewReader(strings.NewReader("id\n\"\"\n")).ReadRecords[struct {
		ID int `df:"id"`
	}](); !errors.Is(err, dataframe.ErrInvalidRecord) {
		t.Fatalf("null scalar error = %v", err)
	}
	if _, err := NewReader(strings.NewReader("other\n1\n")).ReadRecords[struct {
		ID int `df:"id"`
	}](); !errors.Is(err, dataframe.ErrColumnNotFound) {
		t.Fatalf("missing field error = %v", err)
	}
	if _, err := NewReader(strings.NewReader("value\nx\n")).ReadRecords[struct {
		Value complex64 `df:"value"`
	}](); !errors.Is(err, dataframe.ErrUnsupported) {
		t.Fatalf("unsupported field error = %v", err)
	}
	if _, err := NewReader(strings.NewReader("value\nx\n")).ReadRecords[struct {
		Value encoding.TextUnmarshaler `df:"value"`
	}](); !errors.Is(err, dataframe.ErrUnsupported) {
		t.Fatalf("nil TextUnmarshaler interface error = %v", err)
	}
}

func TestWrite(t *testing.T) {
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
	if err := writer.Write(frame); err != nil {
		t.Fatal(err)
	}
	want := "id,name,active\n1,\"a,b\",true\n2,NULL,false\n"
	if got := output.String(); got != want {
		t.Fatalf("CSV = %q, want %q", got, want)
	}

	unsupported, err := dataframe.New(dataframe.Column("value", [][]int{{1}}))
	if err != nil {
		t.Fatal(err)
	}
	if err := Write(&strings.Builder{}, unsupported); !errors.Is(err, dataframe.ErrUnsupported) {
		t.Fatalf("unsupported value error = %v", err)
	}
}

func TestWriteRecordsAndTextMarshaler(t *testing.T) {
	type row struct {
		Code textCode `df:"code"`
		Name *string  `df:"name"`
	}
	name := "A"
	var output strings.Builder
	if err := NewWriter(&output).WriteRecords([]row{{Code: 7, Name: &name}, {Code: 8}}); err != nil {
		t.Fatal(err)
	}
	if got, want := output.String(), "code,name\nC7,A\nC8,\n"; got != want {
		t.Fatalf("CSV = %q, want %q", got, want)
	}
}

func TestWriteRecordsSchemaAndNulls(t *testing.T) {
	type Metadata struct {
		ID int `df:"id"`
	}
	type row struct {
		Metadata
		Name    series.Optional[string] `df:"name"`
		Ignored string                  `df:"-"`
	}
	records := []row{
		{Metadata: Metadata{ID: 1}, Name: series.Some("A")},
		{Metadata: Metadata{ID: 2}},
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

func TestWriteRecordsTreatsNilInterfacesAsPresent(t *testing.T) {
	tests := []struct {
		name  string
		write func(*Writer) error
	}{
		{
			name: "plain",
			write: func(writer *Writer) error {
				return writer.WriteRecords([]struct {
					Value any `df:"value"`
				}{{}})
			},
		},
		{
			name: "optional",
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
		{name: "frame", write: func() error { return Write(&strings.Builder{}, frame) }},
		{name: "records", write: func() error { return NewWriter(&strings.Builder{}).WriteRecords([]struct{}{{}, {}}) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.write(); !errors.Is(err, dataframe.ErrUnsupported) {
				t.Fatalf("error = %v, want ErrUnsupported", err)
			}
		})
	}

	if err := Write(&strings.Builder{}, dataframe.Frame{}); err != nil {
		t.Fatalf("empty frame error = %v", err)
	}
	if err := NewWriter(&strings.Builder{}).WriteRecords([]struct{}{}); err != nil {
		t.Fatalf("empty records error = %v", err)
	}
}

func TestReadWriteConvenienceAndConfiguration(t *testing.T) {
	frame, err := dataframe.New(dataframe.Column("id", []int{1}))
	if err != nil {
		t.Fatal(err)
	}
	var output strings.Builder
	writer := NewWriter(&output)
	writer.Comma = ';'
	writer.UseCRLF = true
	if err := writer.Write(frame); err != nil {
		t.Fatal(err)
	}
	if output.String() != "id\r\n1\r\n" {
		t.Fatalf("configured output = %q", output.String())
	}
	if err := NewWriter(&strings.Builder{}).Write(frame); err != nil {
		t.Fatal(err)
	}
	if err := (*Writer)(nil).Write(frame); err == nil {
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
		if err := writer.Write(frame); err == nil {
			t.Errorf("Write() accepted delimiter %U", delimiter)
		}

		recordWriter := NewWriter(&strings.Builder{})
		recordWriter.Header = false
		recordWriter.Comma = delimiter
		if err := recordWriter.WriteRecords([]struct{ Value int }{}); err == nil {
			t.Errorf("WriteRecords() accepted delimiter %U", delimiter)
		}
	}
}

func BenchmarkRead(b *testing.B) {
	var input strings.Builder
	input.WriteString("id,value\n")
	for i := 0; i < 10_000; i++ {
		fmt.Fprintf(&input, "%d,%f\n", i, float64(i)/3)
	}
	text := input.String()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Read(strings.NewReader(text)); err != nil {
			b.Fatal(err)
		}
	}
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
	for b.Loop() {
		var output strings.Builder
		if err := Write(&output, frame); err != nil {
			b.Fatal(err)
		}
	}
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
	b.ReportAllocs()
	for b.Loop() {
		var output strings.Builder
		if err := NewWriter(&output).WriteRecords(records); err != nil {
			b.Fatal(err)
		}
	}
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

func (c textCode) MarshalText() ([]byte, error) {
	return fmt.Appendf(nil, "C%d", c), nil
}

type incrementingTextCode int

func (c *incrementingTextCode) MarshalText() ([]byte, error) {
	*c = *c + 1
	return fmt.Appendf(nil, "C%d", *c), nil
}
