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
	return []byte(fmt.Sprintf("C%d", c)), nil
}
