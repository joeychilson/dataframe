package dataframe

import (
	"errors"
	"math"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"testing"

	"github.com/joeychilson/dataframe/mask"
	"github.com/joeychilson/dataframe/series"
)

func TestColumnCopiesInput(t *testing.T) {
	values := []int{1, 2}
	frame, err := New(Column("id", values))
	if err != nil {
		t.Fatal(err)
	}
	values[0] = 99
	column, err := frame.Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got := column.Values(); !slices.Equal(got, []int{1, 2}) {
		t.Fatalf("column values = %v", got)
	}
}

func TestNew_PreservesShapeAndSchema(t *testing.T) {
	notes, err := series.NewNullable([]string{"one", ""}, []bool{true, false})
	if err != nil {
		t.Fatal(err)
	}
	frame, err := New(Column("id", []int{1, 2}), ColumnFromSeries("note", notes))
	if err != nil {
		t.Fatal(err)
	}

	if frame.Len() != 2 || frame.Width() != 2 {
		t.Fatalf("shape = %dx%d, want 2x2", frame.Len(), frame.Width())
	}
	if got := frame.Names(); !slices.Equal(got, []string{"id", "note"}) {
		t.Fatalf("Names() = %v", got)
	}
	wantSchema := []Field{
		{Name: "id", Type: reflect.TypeFor[int]()},
		{Name: "note", Type: reflect.TypeFor[string](), Nullable: true},
	}
	if got := frame.Schema(); !reflect.DeepEqual(got, wantSchema) {
		t.Fatalf("Schema() = %#v, want %#v", got, wantSchema)
	}
	if !frame.Has("id") || frame.Has("missing") {
		t.Fatalf("Has returned an unexpected result")
	}
	if got := frame.String(); got != "Frame[2x2]{id:int, note:string?}" {
		t.Fatalf("String() = %q", got)
	}

	// Returned schema slices must not mutate the frame.
	names := frame.Names()
	names[0] = "changed"
	fields := frame.Schema()
	fields[0].Name = "changed"
	if !frame.Has("id") {
		t.Fatal("Names or Schema exposed frame storage")
	}
}

func TestNew_RejectsInvalidSchemas(t *testing.T) {
	tests := []struct {
		name    string
		columns []ColumnSpec
		want    error
	}{
		{name: "rejects a nil column specification", columns: []ColumnSpec{nil}, want: ErrSchemaMismatch},
		{name: "rejects an empty column name", columns: []ColumnSpec{Column("", []int{1})}, want: ErrInvalidName},
		{name: "rejects a duplicate column name", columns: []ColumnSpec{Column("x", []int{1}), Column("x", []int{2})}, want: ErrColumnConflict},
		{name: "rejects mismatched row counts", columns: []ColumnSpec{Column("x", []int{1}), Column("y", []int{1, 2})}, want: ErrRowCount},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := New(test.columns...)
			if !errors.Is(err, test.want) {
				t.Fatalf("New error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestFrameColumnAndWith_EnforceTypesAndRowCounts(t *testing.T) {
	frame, err := New(Column("id", []int{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	ids, err := frame.Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got := ids.Values(); !slices.Equal(got, []int{1, 2}) {
		t.Fatalf("Column = %v", got)
	}
	if _, columnErr := frame.Column[string]("id"); !errors.Is(columnErr, ErrColumnType) {
		t.Fatalf("type error = %v", columnErr)
	}
	if _, columnErr := frame.Column[int]("missing"); !errors.Is(columnErr, ErrColumnNotFound) {
		t.Fatalf("missing error = %v", columnErr)
	}

	frame, err = frame.WithValues("name", []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	frame, err = frame.WithValues("id", []int64{10, 20})
	if err != nil {
		t.Fatal(err)
	}
	if got := frame.Names(); !slices.Equal(got, []string{"id", "name"}) {
		t.Fatalf("names after replacement = %v", got)
	}
	if _, columnErr := frame.Column[int64]("id"); columnErr != nil {
		t.Fatalf("replacement column: %v", columnErr)
	}
	if _, withErr := frame.WithValues("bad", []int{1}); !errors.Is(withErr, ErrRowCount) {
		t.Fatalf("row-count error = %v", withErr)
	}
	if _, withErr := frame.WithValues("", []int{1, 2}); !errors.Is(withErr, ErrInvalidName) {
		t.Fatalf("name error = %v", withErr)
	}
}

func TestFrameSchemaTransforms_PreserveOrderAndRows(t *testing.T) {
	frame, err := New(
		Column("a", []int{1, 2}),
		Column("b", []string{"x", "y"}),
		Column("c", []bool{true, false}),
	)
	if err != nil {
		t.Fatal(err)
	}

	selected, err := frame.Select("c", "a")
	if err != nil {
		t.Fatal(err)
	}
	if got := selected.Names(); !slices.Equal(got, []string{"c", "a"}) {
		t.Fatalf("Select names = %v", got)
	}
	empty, err := frame.Select()
	if err != nil {
		t.Fatal(err)
	}
	if empty.Width() != 0 || empty.Len() != 2 {
		t.Fatalf("empty selection shape = %dx%d", empty.Len(), empty.Width())
	}

	renamed, err := frame.Rename("b", "name")
	if err != nil {
		t.Fatal(err)
	}
	if got := renamed.Names(); !slices.Equal(got, []string{"a", "name", "c"}) {
		t.Fatalf("Rename names = %v", got)
	}
	if _, renameErr := frame.Rename("a", "b"); !errors.Is(renameErr, ErrColumnConflict) {
		t.Fatalf("rename conflict = %v", renameErr)
	}
	if _, renameErr := frame.Rename("missing", "name"); !errors.Is(renameErr, ErrColumnNotFound) {
		t.Fatalf("rename missing error = %v", renameErr)
	}
	if _, renameErr := frame.Rename("a", ""); !errors.Is(renameErr, ErrInvalidName) {
		t.Fatalf("rename empty error = %v", renameErr)
	}
	unchanged, err := frame.Rename("a", "a")
	if err != nil || !slices.Equal(unchanged.Names(), frame.Names()) {
		t.Fatalf("same-name Rename = %v, %v", unchanged.Names(), err)
	}

	dropped, err := frame.Drop("a", "c", "a")
	if err != nil {
		t.Fatal(err)
	}
	if got := dropped.Names(); !slices.Equal(got, []string{"b"}) {
		t.Fatalf("Drop names = %v", got)
	}
	if _, dropErr := frame.Drop("missing"); !errors.Is(dropErr, ErrColumnNotFound) {
		t.Fatalf("drop error = %v", dropErr)
	}
	if _, selectErr := frame.Select("a", "a"); !errors.Is(selectErr, ErrColumnConflict) {
		t.Fatalf("duplicate select error = %v", selectErr)
	}
	if _, selectErr := frame.Select("missing"); !errors.Is(selectErr, ErrColumnNotFound) {
		t.Fatalf("missing select error = %v", selectErr)
	}
}

func TestFrameRowTransforms_SelectExpectedRows(t *testing.T) {
	frame, err := New(Column("n", []int{10, 20, 30, 40}))
	if err != nil {
		t.Fatal(err)
	}
	if identity := frame.Take([]int{0, 1, 2, 3}); &identity.columns[0] != &frame.columns[0] {
		t.Fatal("Take(identity) copied an immutable Frame")
	}
	if identity := frame.Filter(mask.All(frame.Len())); &identity.columns[0] != &frame.columns[0] {
		t.Fatal("Filter(all) copied an immutable Frame")
	}
	for _, identity := range []struct {
		name  string
		frame Frame
	}{
		{name: "Head reuses a full-range frame", frame: frame.Head(frame.Len() + 1)},
		{name: "Tail reuses a full-range frame", frame: frame.Tail(frame.Len() + 1)},
		{name: "Slice reuses a full-range frame", frame: frame.Slice(0, frame.Len())},
	} {
		if &identity.frame.columns[0] != &frame.columns[0] {
			t.Errorf("%s; copied an immutable Frame", identity.name)
		}
	}

	assertInts := func(t *testing.T, gotFrame Frame, want []int) {
		t.Helper()
		values, columnErr := gotFrame.Column[int]("n")
		if columnErr != nil {
			t.Fatal(columnErr)
		}
		if got := values.Values(); !slices.Equal(got, want) {
			t.Fatalf("values = %v, want %v", got, want)
		}
	}
	assertInts(t, frame.Take([]int{3, 1, 1}), []int{40, 20, 20})
	assertInts(t, frame.Head(2), []int{10, 20})
	assertInts(t, frame.Tail(2), []int{30, 40})
	assertInts(t, frame.Slice(1, 3), []int{20, 30})
	assertInts(t, frame.Filter(mask.New([]bool{true, false, true, false})), []int{10, 30})
	assertInts(t, frame.FilterFunc(series.New([]int{1, 2, 3, 4}), func(value int) bool {
		return value%2 == 0
	}), []int{20, 40})

	zeroWidth, err := frame.Select()
	if err != nil {
		t.Fatal(err)
	}
	if got := zeroWidth.Take([]int{3, 0, 0}); got.Len() != 3 || got.Width() != 0 {
		t.Fatalf("zero-width Take shape = %dx%d", got.Len(), got.Width())
	}
	invalid := []struct {
		name string
		call func()
	}{
		{name: "Take rejects a negative index", call: func() { frame.Take([]int{-1}) }},
		{name: "Take rejects an index equal to length", call: func() { zeroWidth.Take([]int{4}) }},
		{name: "Head rejects a negative count", call: func() { frame.Head(-1) }},
		{name: "Tail rejects a negative count", call: func() { frame.Tail(-1) }},
		{name: "Slice rejects a negative bound", call: func() { frame.Slice(-1, 1) }},
		{name: "Slice rejects reversed bounds", call: func() { frame.Slice(2, 1) }},
		{name: "Slice rejects an end past length", call: func() { frame.Slice(0, 5) }},
		{name: "Filter rejects a mismatched length", call: func() { frame.Filter(mask.All(3)) }},
		{name: "FilterFunc rejects a mismatched length", call: func() { frame.FilterFunc(series.New([]int{1}), func(int) bool { return true }) }},
		{name: "FilterFunc rejects a nil predicate", call: func() {
			frame.FilterFunc(series.FromOptionals(make([]series.Optional[int], frame.Len())), nil)
		}},
	}
	for _, test := range invalid {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("function did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestFrameConcat_AppendsCompatibleRows(t *testing.T) {
	left, err := New(Column("id", []int{1}), Column("name", []string{"a"}))
	if err != nil {
		t.Fatal(err)
	}
	nullable, err := series.NewNullable([]string{"", "c"}, []bool{false, true})
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("id", []int{2, 3}), ColumnFromSeries("name", nullable))
	if err != nil {
		t.Fatal(err)
	}
	joined, err := left.Concat(right)
	if err != nil {
		t.Fatal(err)
	}
	if joined.Len() != 3 || !joined.Schema()[1].Nullable {
		t.Fatalf("Concat shape/schema = %dx%d %#v", joined.Len(), joined.Width(), joined.Schema())
	}
	names, err := joined.Column[string]("name")
	if err != nil {
		t.Fatal(err)
	}
	if got := names.Validity(); !slices.Equal(got, []bool{true, false, true}) {
		t.Fatalf("Concat validity = %v", got)
	}

	wrongWidth, err := New(Column("id", []int{1}))
	if err != nil {
		t.Fatal(err)
	}
	wrongName, err := New(Column("id", []int{1}), Column("label", []string{"x"}))
	if err != nil {
		t.Fatal(err)
	}
	wrongType, err := New(Column("id", []int64{1}), Column("name", []string{"x"}))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name  string
		other Frame
	}{
		{name: "rejects a mismatched width", other: wrongWidth},
		{name: "rejects a mismatched column name", other: wrongName},
		{name: "rejects a mismatched column type", other: wrongType},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, concatErr := left.Concat(test.other); !errors.Is(concatErr, ErrSchemaMismatch) {
				t.Fatalf("Concat error = %v, want ErrSchemaMismatch", concatErr)
			}
		})
	}

	zeroWidth, err := (Frame{rowCount: 2}).Concat(Frame{rowCount: 3})
	if err != nil {
		t.Fatal(err)
	}
	if zeroWidth.Len() != 5 || zeroWidth.Width() != 0 {
		t.Fatalf("zero-width Concat shape = %dx%d, want 5x0", zeroWidth.Len(), zeroWidth.Width())
	}
}

func TestConcatRowCountOverflowReturnsError(t *testing.T) {
	_, err := (Frame{rowCount: math.MaxInt}).Concat(Frame{rowCount: 1})
	if !errors.Is(err, ErrRowCount) {
		t.Fatalf("Concat() overflow error = %v, want ErrRowCount", err)
	}
}

func BenchmarkNewFrame(b *testing.B) {
	const width = 16
	columns := make([]ColumnSpec, width)
	for i := range columns {
		columns[i] = ColumnFromSeries("column"+strconv.Itoa(i), series.Repeat(i, 100))
	}
	b.ReportAllocs()
	var result Frame
	for b.Loop() {
		var err error
		result, err = New(columns...)
		if err != nil {
			b.Fatal(err)
		}
	}
	runtime.KeepAlive(result)
}

func BenchmarkFrameFilter(b *testing.B) {
	values := make([]int, 10_000)
	frame, err := New(Column("value", values))
	if err != nil {
		b.Fatal(err)
	}
	implementations := []struct {
		name   string
		filter func(mask.Mask) Frame
	}{
		{name: "Optimized", filter: frame.Filter},
		{name: "Reference", filter: func(selection mask.Mask) Frame {
			rowCount := selection.Count()
			columns := make([]column, len(frame.columns))
			for i, stored := range frame.columns {
				columns[i] = stored
				columns[i].values = stored.values.filter(selection)
			}
			return Frame{columns: columns, rowCount: rowCount}
		}},
	}
	for _, benchmark := range []struct {
		name      string
		selection mask.Mask
	}{
		{name: "All", selection: mask.All(len(values))},
		{name: "Half", selection: mask.NewFunc(len(values), func(i int) bool { return i%2 == 0 })},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			for _, implementation := range implementations {
				b.Run(implementation.name, func(b *testing.B) {
					b.ReportAllocs()
					var result Frame
					for b.Loop() {
						result = implementation.filter(benchmark.selection)
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}

func BenchmarkFrameTake(b *testing.B) {
	const size = 10_000
	frame, err := New(
		Column("id", make([]int, size)),
		Column("name", make([]string, size)),
		Column("value", make([]float64, size)),
	)
	if err != nil {
		b.Fatal(err)
	}
	identity := make([]int, size)
	for i := range identity {
		identity[i] = i
	}
	subset := make([]int, size/2)
	for i := range subset {
		subset[i] = size - 1 - i*2
	}
	implementations := []struct {
		name string
		take func([]int) Frame
	}{
		{name: "Optimized", take: frame.Take},
		{name: "Reference", take: func(rows []int) Frame {
			columns := make([]column, len(frame.columns))
			for i, stored := range frame.columns {
				columns[i] = stored
				columns[i].values = stored.values.take(rows)
			}
			return Frame{columns: columns, rowCount: len(rows)}
		}},
	}
	for _, benchmark := range []struct {
		name string
		rows []int
	}{
		{name: "Identity", rows: identity},
		{name: "Subset", rows: subset},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			for _, implementation := range implementations {
				b.Run(implementation.name, func(b *testing.B) {
					b.ReportAllocs()
					var result Frame
					for b.Loop() {
						result = implementation.take(benchmark.rows)
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}

func BenchmarkFrameSlice(b *testing.B) {
	const size = 10_000
	for _, width := range []int{0, 1, 16, 64} {
		b.Run(strconv.Itoa(width)+"-columns", func(b *testing.B) {
			columns := make([]ColumnSpec, width)
			for i := range columns {
				columns[i] = ColumnFromSeries(strconv.Itoa(i), series.Repeat(i, size))
			}
			frame, err := New(columns...)
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			var result Frame
			for b.Loop() {
				result = frame.Slice(0, frame.Len())
			}
			runtime.KeepAlive(result)
		})
	}
}

func BenchmarkFrameConcat(b *testing.B) {
	const size = 2_500
	parts := make([]Frame, 4)
	for i := range parts {
		var err error
		parts[i], err = New(
			Column("id", make([]int, size)),
			Column("name", make([]string, size)),
			Column("value", make([]float64, size)),
		)
		if err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	var result Frame
	for b.Loop() {
		var err error
		result, err = parts[0].Concat(parts[1:]...)
		if err != nil {
			b.Fatal(err)
		}
	}
	runtime.KeepAlive(result)
}
