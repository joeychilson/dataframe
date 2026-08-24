package dataframe

import (
	"errors"
	"hash/maphash"
	"math"
	"reflect"
	"runtime"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/internal/bitmap"
	"github.com/joeychilson/dataframe/series"
)

func TestFrameDistinct_KeepsFirstOccurrences(t *testing.T) {
	names, err := series.NewNullable([]string{"a", "a", "", "", "b"}, []bool{true, true, false, false, true})
	if err != nil {
		t.Fatal(err)
	}
	frame, err := New(Column("n", []int{1, 1, 1, 1, 2}), ColumnFromSeries("name", names))
	if err != nil {
		t.Fatal(err)
	}
	distinct, err := frame.Distinct()
	if err != nil {
		t.Fatal(err)
	}
	if distinct.Len() != 3 {
		t.Fatalf("Distinct length = %d, want 3", distinct.Len())
	}

	by := frame.DistinctBy(series.New([]int{1, 1, 2, 2, 1}))
	if got, _ := by.Column[int]("n"); !slices.Equal(got.Values(), []int{1, 1}) {
		t.Fatalf("DistinctBy values = %v", got.Values())
	}
	using := frame.DistinctByUsing(series.New([]int{1, 1, 2, 2, 1}), maphash.ComparableHasher[int]{})
	if using.Len() != 2 {
		t.Fatalf("DistinctByUsing length = %d", using.Len())
	}
	zeroWidth, err := (Frame{rowCount: 3}).Distinct()
	if err != nil {
		t.Fatal(err)
	}
	if zeroWidth.Len() != 1 || zeroWidth.Width() != 0 {
		t.Fatalf("zero-width Distinct shape = %dx%d, want 1x0", zeroWidth.Len(), zeroWidth.Width())
	}

	unsupported, err := New(Column("slice", [][]int{{1}, {1}}))
	if err != nil {
		t.Fatal(err)
	}
	if _, distinctErr := unsupported.Distinct(); !errors.Is(distinctErr, ErrUnsupported) {
		t.Fatalf("Distinct error = %v", distinctErr)
	}
	dynamic, err := New(Column[any]("dynamic", []any{[]int{1}, []int{1}}))
	if err != nil {
		t.Fatal(err)
	}
	if _, distinctErr := dynamic.Distinct(); !errors.Is(distinctErr, ErrUnsupported) {
		t.Fatalf("dynamic Distinct error = %v", distinctErr)
	}

	type nestedDynamic struct {
		Value any
	}
	nested, err := New(Column("nested", []nestedDynamic{{Value: []int{1}}, {Value: []int{1}}}))
	if err != nil {
		t.Fatal(err)
	}
	if _, distinctErr := nested.Distinct(); !errors.Is(distinctErr, ErrUnsupported) {
		t.Fatalf("nested dynamic Distinct error = %v", distinctErr)
	}
}

func TestFrameDistinctSeparatesPresentNilAndNull(t *testing.T) {
	values, err := series.NewNullable([]any{nil, nil, nil}, []bool{true, false, true})
	if err != nil {
		t.Fatal(err)
	}
	frame, err := New(ColumnFromSeries("value", values))
	if err != nil {
		t.Fatal(err)
	}
	distinct, err := frame.Distinct()
	if err != nil {
		t.Fatal(err)
	}
	if got := distinct.Len(); got != 2 {
		t.Fatalf("Distinct length = %d, want 2", got)
	}
	distinctValues, err := distinct.Column[any]("value")
	if err != nil {
		t.Fatal(err)
	}
	if got := distinctValues.Validity(); !slices.Equal(got, []bool{true, false}) {
		t.Fatalf("Distinct validity = %v", got)
	}
}

func TestFrameDistinctSingleColumnPreservesOrderAndNulls(t *testing.T) {
	values, err := series.NewNullable(
		[]int{2, 1, 2, 0, 1, 0},
		[]bool{true, true, true, false, true, false},
	)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := New(ColumnFromSeries("value", values))
	if err != nil {
		t.Fatal(err)
	}
	distinct, err := frame.Distinct()
	if err != nil {
		t.Fatal(err)
	}
	distinctValues, err := distinct.Column[int]("value")
	if err != nil {
		t.Fatal(err)
	}
	want := []series.Optional[int]{series.Some(2), series.Some(1), series.None[int]()}
	if got := distinctValues.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("Distinct values = %v, want %v", got, want)
	}
}

func TestFrameDistinctSingleDefinedColumnUsesFallback(t *testing.T) {
	type code int
	frame, err := New(Column("code", []code{2, 1, 2, 3, 1}))
	if err != nil {
		t.Fatal(err)
	}
	distinct, err := frame.Distinct()
	if err != nil {
		t.Fatal(err)
	}
	values, err := distinct.Column[code]("code")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := values.Values(), []code{2, 1, 3}; !slices.Equal(got, want) {
		t.Fatalf("Distinct values = %v, want %v", got, want)
	}
}

func TestFrameDistinctMultipleColumnsPreservesOrderAndNulls(t *testing.T) {
	left, err := series.NewNullable(
		[]int{2, 2, 0, 0, 2, 2, 1, 1},
		[]bool{true, true, false, false, true, true, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	right, err := series.NewNullable(
		[]string{"a", "a", "", "", "", "", "a", "a"},
		[]bool{true, true, false, false, false, false, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := New(ColumnFromSeries("left", left), ColumnFromSeries("right", right))
	if err != nil {
		t.Fatal(err)
	}

	distinct, err := frame.Distinct()
	if err != nil {
		t.Fatal(err)
	}
	distinctLeft, err := distinct.Column[int]("left")
	if err != nil {
		t.Fatal(err)
	}
	distinctRight, err := distinct.Column[string]("right")
	if err != nil {
		t.Fatal(err)
	}
	wantLeft := []series.Optional[int]{series.Some(2), series.None[int](), series.Some(2), series.Some(1)}
	if got := distinctLeft.Optionals(); !slices.Equal(got, wantLeft) {
		t.Fatalf("Distinct left values = %v, want %v", got, wantLeft)
	}
	wantRight := []series.Optional[string]{series.Some("a"), series.None[string](), series.None[string](), series.Some("a")}
	if got := distinctRight.Optionals(); !slices.Equal(got, wantRight) {
		t.Fatalf("Distinct right values = %v, want %v", got, wantRight)
	}
}

func TestFrameDistinctMultipleColumnsPreservesFloatEquality(t *testing.T) {
	frame, err := New(
		Column("value", []float64{0, math.Copysign(0, -1), math.NaN(), math.NaN()}),
		Column("group", []int{1, 1, 1, 1}),
	)
	if err != nil {
		t.Fatal(err)
	}
	distinct, err := frame.Distinct()
	if err != nil {
		t.Fatal(err)
	}
	if got := distinct.Len(); got != 3 {
		t.Fatalf("Distinct length = %d, want 3", got)
	}
	values, err := distinct.Column[float64]("value")
	if err != nil {
		t.Fatal(err)
	}
	got := values.Values()
	if math.Signbit(got[0]) || !math.IsNaN(got[1]) || !math.IsNaN(got[2]) {
		t.Fatalf("Distinct values = %v, want positive zero followed by two NaNs", got)
	}
}

func TestFrameDistinctMultipleColumnsDistributesSelfUnequalValues(t *testing.T) {
	const size = 2_048
	values := slices.Repeat([]float64{math.Float64frombits(0x7ff8_0000_0000_0001)}, size)
	frame, err := New(Column("value", values), Column("group", make([]int, size)))
	if err != nil {
		t.Fatal(err)
	}
	distinct, err := frame.Distinct()
	if err != nil {
		t.Fatal(err)
	}
	if distinct.Len() != size {
		t.Fatalf("Distinct length = %d, want %d", distinct.Len(), size)
	}
}

func TestFrameDistinctMultipleColumnsUsesFallback(t *testing.T) {
	type code int
	frame, err := New(
		Column("value", []int{2, 1, 2, 3, 1}),
		Column("code", []code{20, 10, 20, 30, 10}),
	)
	if err != nil {
		t.Fatal(err)
	}
	distinct, err := frame.Distinct()
	if err != nil {
		t.Fatal(err)
	}
	values, err := distinct.Column[int]("value")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := values.Values(), []int{2, 1, 3}; !slices.Equal(got, want) {
		t.Fatalf("Distinct values = %v, want %v", got, want)
	}
}

func TestFrameDistinctByPanicsOnInvalidArguments(t *testing.T) {
	frame, err := New(Column("value", []int{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		call func()
	}{
		{name: "DistinctBy rejects a mismatched key length", call: func() { frame.DistinctBy(series.New([]int{1})) }},
		{name: "DistinctByUsing rejects a mismatched key length", call: func() { frame.DistinctByUsing(series.New([]int{1}), maphash.ComparableHasher[int]{}) }},
		{name: "DistinctByUsing rejects a nil hasher", call: func() { frame.DistinctByUsing(series.New([]int{1, 2}), nil) }},
	}
	for _, test := range tests {
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

func BenchmarkFrameDistinct(b *testing.B) {
	const size = 10_000
	type benchmark struct {
		name  string
		frame Frame
	}
	var benchmarks []benchmark
	for _, cardinality := range []int{100, size} {
		label := "100-unique"
		if cardinality == size {
			label = "all-unique"
		}
		left := make([]int, size)
		right := make([]int, size)
		for i := range left {
			left[i] = i % cardinality
			right[i] = (i * 31) % cardinality
		}
		single, err := New(Column("value", left))
		if err != nil {
			b.Fatal(err)
		}
		multiple, err := New(Column("left", left), Column("right", right))
		if err != nil {
			b.Fatal(err)
		}
		benchmarks = append(benchmarks,
			benchmark{name: "single/" + label, frame: single},
			benchmark{name: "multiple/" + label, frame: multiple},
		)
	}
	allNaN, err := New(
		Column("value", slices.Repeat([]float64{math.Float64frombits(0x7ff8_0000_0000_0001)}, size)),
		Column("group", make([]int, size)),
	)
	if err != nil {
		b.Fatal(err)
	}
	benchmarks = append(benchmarks, benchmark{name: "multiple/all-nan", frame: allNaN})
	nullable := series.NewNullableFunc(size, func(i int) (int, bool) { return i % 100, i%4 != 0 })
	nullableFrame, err := New(ColumnFromSeries("value", nullable))
	if err != nil {
		b.Fatal(err)
	}
	benchmarks = append(benchmarks, benchmark{name: "single/nullable", frame: nullableFrame})
	for _, benchmark := range benchmarks {
		reflected := Frame{columns: slices.Clone(benchmark.frame.columns), rowCount: benchmark.frame.Len()}
		for i, stored := range benchmark.frame.columns {
			values := reflect.MakeSlice(reflect.SliceOf(stored.typeOf), benchmark.frame.Len(), benchmark.frame.Len())
			validity := make([]bool, benchmark.frame.Len())
			for row := range benchmark.frame.Len() {
				value, present := stored.values.at(row)
				validity[row] = present
				if present {
					values.Index(row).Set(reflect.ValueOf(value))
				}
			}
			data := reflectData{values: values}
			if stored.nullable {
				data.validity = bitmap.FromBools(validity)
			}
			reflected.columns[i].values = data
		}
		b.Run(benchmark.name, func(b *testing.B) {
			for _, implementation := range []struct {
				name  string
				frame Frame
			}{
				{name: "Optimized", frame: benchmark.frame},
				{name: "Reference", frame: reflected},
			} {
				b.Run(implementation.name, func(b *testing.B) {
					b.ReportAllocs()
					var result Frame
					for b.Loop() {
						var distinctErr error
						result, distinctErr = implementation.frame.Distinct()
						if distinctErr != nil {
							b.Fatal(distinctErr)
						}
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}

func BenchmarkFrameDistinctBy(b *testing.B) {
	values := make([]int, 10_000)
	for i := range values {
		values[i] = i % 100
	}
	frame, err := New(Column("value", values))
	if err != nil {
		b.Fatal(err)
	}
	key := series.New(values)
	b.ReportAllocs()
	var result Frame
	for b.Loop() {
		result = frame.DistinctBy(key)
	}
	runtime.KeepAlive(result)
}
