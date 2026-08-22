package series

import (
	"errors"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/mask"
)

func TestNew(t *testing.T) {
	input := []int{10, 20, 30}
	values := New(input)
	input[0] = 99

	if values.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", values.Len())
	}
	if values.Nullable() {
		t.Fatal("Nullable() = true, want false")
	}
	if values.NullCount() != 0 {
		t.Fatalf("NullCount() = %d, want 0", values.NullCount())
	}
	if got := values.Values(); !slices.Equal(got, []int{10, 20, 30}) {
		t.Fatalf("Values() = %v, want [10 20 30]", got)
	}
	if got := values.Validity(); !slices.Equal(got, []bool{true, true, true}) {
		t.Fatalf("Validity() = %v, want [true true true]", got)
	}
}

func BenchmarkNew(b *testing.B) {
	input := make([]int, 1024)
	b.ReportAllocs()
	b.ResetTimer()

	var result Series[int]
	for b.Loop() {
		result = New(input)
	}
	runtime.KeepAlive(result)
}

func TestNewNullable(t *testing.T) {
	input := []string{"first", "hidden", "third"}
	validity := []bool{true, false, true}
	values, err := NewNullable(input, validity)
	if err != nil {
		t.Fatalf("NewNullable() error = %v", err)
	}

	input[0] = "changed"
	validity[0] = false

	physical := values.Values()
	if physical[0] != "first" || physical[2] != "third" {
		t.Fatalf("Values() at present rows = %v, want copied input", physical)
	}
	if got := values.Validity(); !slices.Equal(got, []bool{true, false, true}) {
		t.Fatalf("Validity() = %v, want copied validity", got)
	}
	if values.NullCount() != 1 {
		t.Fatalf("NullCount() = %d, want 1", values.NullCount())
	}
}

func BenchmarkNewNullable(b *testing.B) {
	input := make([]int, 1024)
	validity := slices.Repeat([]bool{true}, len(input))
	b.ReportAllocs()
	b.ResetTimer()

	var result Series[int]
	for b.Loop() {
		var err error
		result, err = NewNullable(input, validity)
		if err != nil {
			b.Fatal(err)
		}
	}
	runtime.KeepAlive(result)
}

func TestNewNullablePreservesNullableSchema(t *testing.T) {
	tests := []struct {
		name   string
		values []int
		valid  []bool
	}{
		{name: "all present", values: []int{1, 2}, valid: []bool{true, true}},
		{name: "empty", values: nil, valid: nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			values, err := NewNullable(test.values, test.valid)
			if err != nil {
				t.Fatalf("NewNullable() error = %v", err)
			}
			if !values.Nullable() {
				t.Fatal("Nullable() = false, want true")
			}
		})
	}
}

func TestNewNullableLengthMismatch(t *testing.T) {
	values, err := NewNullable([]int{1, 2}, []bool{true})
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("NewNullable() error = %v, want ErrLengthMismatch", err)
	}
	if values.Len() != 0 {
		t.Fatalf("NewNullable() Series length = %d, want 0", values.Len())
	}
}

func TestFromOptionals(t *testing.T) {
	input := []Optional[int]{Some(10), {Value: 99}, Some(30)}
	values := FromOptionals(input)
	input[0] = None[int]()

	if got := values.Optionals(); !slices.Equal(got, []Optional[int]{Some(10), None[int](), Some(30)}) {
		t.Fatalf("Optionals() = %+v, want present, null, present", got)
	}
	if !values.Nullable() {
		t.Fatal("Nullable() = false, want true")
	}
}

func TestFromOptionalsPreservesNullableSchema(t *testing.T) {
	tests := []struct {
		name   string
		values []Optional[int]
	}{
		{name: "all present", values: []Optional[int]{Some(1), Some(2)}},
		{name: "empty", values: nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if values := FromOptionals(test.values); !values.Nullable() {
				t.Fatal("Nullable() = false, want true")
			}
		})
	}
}

func TestRepeat(t *testing.T) {
	values := Repeat("go", 3)
	if got := values.Values(); !slices.Equal(got, []string{"go", "go", "go"}) {
		t.Fatalf("Repeat(\"go\", 3).Values() = %v", got)
	}
	if values.NullCount() != 0 {
		t.Fatalf("NullCount() = %d, want 0", values.NullCount())
	}
	if values.Nullable() {
		t.Fatal("Nullable() = true, want false")
	}
}

func TestRepeatPanicsForNegativeCount(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Repeat(value, -1) did not panic")
		}
	}()
	Repeat(1, -1)
}

func TestNullable(t *testing.T) {
	var values Series[int]
	if values.Nullable() {
		t.Fatal("zero Series Nullable() = true, want false")
	}
}

func TestSeriesAccess(t *testing.T) {
	values, err := NewNullable(
		[]string{"first", "physical null value", "third"},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatalf("NewNullable() error = %v", err)
	}

	if value, ok := values.At(0); value != "first" || !ok {
		t.Fatalf("At(0) = (%q, %t), want (\"first\", true)", value, ok)
	}
	if value, ok := values.At(1); value != "" || ok {
		t.Fatalf("At(1) = (%q, %t), want (\"\", false)", value, ok)
	}
	if !values.IsValid(0) || values.IsValid(1) {
		t.Fatalf("IsValid() = [%t %t], want [true false]", values.IsValid(0), values.IsValid(1))
	}
	if value := values.ValueOr(0, "fallback"); value != "first" {
		t.Fatalf("ValueOr(0) = %q, want \"first\"", value)
	}
	if value := values.ValueOr(1, "fallback"); value != "fallback" {
		t.Fatalf("ValueOr(1) = %q, want \"fallback\"", value)
	}
}

func TestSeriesAccessPanicsOutOfRange(t *testing.T) {
	values := New([]int{1})
	tests := []struct {
		name string
		call func()
	}{
		{name: "IsValid negative", call: func() { values.IsValid(-1) }},
		{name: "IsValid length", call: func() { values.IsValid(values.Len()) }},
		{name: "At negative", call: func() { values.At(-1) }},
		{name: "At length", call: func() { values.At(values.Len()) }},
		{name: "ValueOr negative", call: func() { values.ValueOr(-1, 0) }},
		{name: "ValueOr length", call: func() { values.ValueOr(values.Len(), 0) }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("call did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestFirstAndLastPresent(t *testing.T) {
	tests := []struct {
		name      string
		values    Series[int]
		wantFirst int
		wantLast  int
		wantOK    bool
	}{
		{name: "zero value", values: Series[int]{}},
		{name: "non-null", values: New([]int{10, 20, 30}), wantFirst: 10, wantLast: 30, wantOK: true},
		{name: "nullable", values: FromOptionals([]Optional[int]{None[int](), Some(20), None[int](), Some(40)}), wantFirst: 20, wantLast: 40, wantOK: true},
		{name: "all null", values: FromOptionals([]Optional[int]{None[int](), None[int]()})},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			first, firstOK := test.values.FirstPresent()
			if first != test.wantFirst || firstOK != test.wantOK {
				t.Errorf("FirstPresent() = (%d, %t), want (%d, %t)", first, firstOK, test.wantFirst, test.wantOK)
			}
			last, lastOK := test.values.LastPresent()
			if last != test.wantLast || lastOK != test.wantOK {
				t.Errorf("LastPresent() = (%d, %t), want (%d, %t)", last, lastOK, test.wantLast, test.wantOK)
			}
		})
	}
}

func TestSeriesSnapshotsAreIndependent(t *testing.T) {
	values := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})

	physical := values.Values()
	validity := values.Validity()
	optionals := values.Optionals()
	physical[0] = 100
	validity[0] = false
	optionals[0] = None[int]()

	if value, ok := values.At(0); value != 10 || !ok {
		t.Fatalf("At(0) after mutating snapshots = (%d, %t), want (10, true)", value, ok)
	}
}

func TestSeriesImmutabilityIsShallow(t *testing.T) {
	referenced := []int{10}
	input := [][]int{referenced}
	values := New(input)

	input[0] = []int{20}
	value, _ := values.At(0)
	if value[0] != 10 {
		t.Fatalf("replacing input element changed Series value to %v", value)
	}

	referenced[0] = 30
	value, _ = values.At(0)
	if value[0] != 30 {
		t.Fatalf("referenced mutation did not reach Series value: %v", value)
	}

	snapshot := values.Values()
	snapshot[0][0] = 40
	value, _ = values.At(0)
	if value[0] != 40 {
		t.Fatalf("referenced snapshot mutation did not reach Series value: %v", value)
	}

	snapshot[0] = []int{50}
	value, _ = values.At(0)
	if value[0] != 40 {
		t.Fatalf("replacing snapshot element changed Series value to %v", value)
	}
}

func TestAll(t *testing.T) {
	values := FromOptionals([]Optional[string]{Some("first"), None[string](), Some("third")})

	var indexes []int
	var cells []Optional[string]
	for index, cell := range values.All() {
		indexes = append(indexes, index)
		cells = append(cells, cell)
	}

	if !slices.Equal(indexes, []int{0, 1, 2}) {
		t.Fatalf("All() indexes = %v, want [0 1 2]", indexes)
	}
	if !slices.Equal(cells, []Optional[string]{Some("first"), None[string](), Some("third")}) {
		t.Fatalf("All() cells = %+v, want present, null, present", cells)
	}

	yields := 0
	values.All()(func(int, Optional[string]) bool {
		yields++
		return false
	})
	if yields != 1 {
		t.Fatalf("All() yielded %d rows after false, want 1", yields)
	}
}

func TestPresent(t *testing.T) {
	values := FromOptionals([]Optional[int]{None[int](), Some(20), None[int](), Some(40)})

	var indexes []int
	var present []int
	for index, value := range values.Present() {
		indexes = append(indexes, index)
		present = append(present, value)
	}

	if !slices.Equal(indexes, []int{1, 3}) {
		t.Fatalf("Present() indexes = %v, want [1 3]", indexes)
	}
	if !slices.Equal(present, []int{20, 40}) {
		t.Fatalf("Present() values = %v, want [20 40]", present)
	}

	yields := 0
	values.Present()(func(int, int) bool {
		yields++
		return false
	})
	if yields != 1 {
		t.Fatalf("Present() yielded %d rows after false, want 1", yields)
	}
}

func BenchmarkPresent(b *testing.B) {
	input := make([]Optional[int], 1024)
	for i := range input {
		if i%4 != 0 {
			input[i] = Some(i)
		}
	}
	allPresent, err := NewNullable(
		slices.Repeat([]int{1}, len(input)),
		slices.Repeat([]bool{true}, len(input)),
	)
	if err != nil {
		b.Fatal(err)
	}
	benchmarks := []struct {
		name   string
		values Series[int]
	}{
		{name: "non-null", values: Repeat(1, len(input))},
		{name: "nullable/all-present", values: allPresent},
		{name: "nullable/25%-null", values: FromOptionals(input)},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			total := 0
			for b.Loop() {
				for _, value := range benchmark.values.Present() {
					total += value
				}
			}
			runtime.KeepAlive(total)
		})
	}
}

func TestFilter(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30), None[int]()})
	tests := []struct {
		name      string
		selection mask.Mask
		want      []Optional[int]
	}{
		{
			name:      "with null",
			selection: mask.New([]bool{true, true, false, false}),
			want:      []Optional[int]{Some(10), None[int]()},
		},
		{
			name:      "present rows only",
			selection: mask.New([]bool{true, false, true, false}),
			want:      []Optional[int]{Some(10), Some(30)},
		},
		{
			name:      "empty result",
			selection: mask.None(source.Len()),
			want:      nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			filtered := source.Filter(test.selection)
			if got := filtered.Optionals(); !slices.Equal(got, test.want) {
				t.Fatalf("Filter() = %+v, want %+v", got, test.want)
			}
			if !filtered.Nullable() {
				t.Fatal("Filter() removed nullable schema")
			}
		})
	}

	dense := New([]int{10, 20, 30}).Filter(mask.New([]bool{true, false, true}))
	if dense.Nullable() {
		t.Fatal("Filter() made a non-null Series nullable")
	}
	if got := dense.Values(); !slices.Equal(got, []int{10, 30}) {
		t.Fatalf("non-null Filter() = %v, want [10 30]", got)
	}
	if got := source.Optionals(); !slices.Equal(got, []Optional[int]{Some(10), None[int](), Some(30), None[int]()}) {
		t.Fatalf("Filter() changed source to %+v", got)
	}
}

func TestFilterPanicsOnLengthMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Filter did not panic")
		}
	}()
	New([]int{1, 2}).Filter(mask.All(1))
}

func BenchmarkFilter(b *testing.B) {
	const length = 1 << 16
	values := make([]int, length)
	validity := make([]bool, length)
	sparseSelection := make([]bool, length)
	for i := range values {
		values[i] = i
		validity[i] = i%4 != 0
		sparseSelection[i] = i%16 == 0
	}
	nullable, err := NewNullable(values, validity)
	if err != nil {
		b.Fatal(err)
	}
	benchmarks := []struct {
		name      string
		values    Series[int]
		selection mask.Mask
	}{
		{name: "non-null/dense", values: New(values), selection: mask.All(length)},
		{name: "non-null/sparse", values: New(values), selection: mask.New(sparseSelection)},
		{name: "nullable/dense", values: nullable, selection: mask.All(length)},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			var result Series[int]
			for b.Loop() {
				result = benchmark.values.Filter(benchmark.selection)
			}
			runtime.KeepAlive(result)
		})
	}
}

func TestTake(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	taken := source.Take([]int{2, 1, 2, 0})
	want := []Optional[int]{Some(30), None[int](), Some(30), Some(10)}
	if got := taken.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("Take() = %+v, want %+v", got, want)
	}
	if !taken.Nullable() {
		t.Fatal("Take() removed nullable schema")
	}

	empty := source.Take(nil)
	if empty.Len() != 0 || !empty.Nullable() {
		t.Fatalf("empty Take() = {Len:%d Nullable:%t}", empty.Len(), empty.Nullable())
	}

	dense := New([]int{10, 20, 30}).Take([]int{2, 0})
	if dense.Nullable() {
		t.Fatal("Take() made a non-null Series nullable")
	}
	if got := dense.Values(); !slices.Equal(got, []int{30, 10}) {
		t.Fatalf("non-null Take() = %v, want [30 10]", got)
	}
}

func TestTakePanicsOnInvalidIndex(t *testing.T) {
	values := New([]int{10, 20})
	tests := []struct {
		name string
		rows []int
	}{
		{name: "negative", rows: []int{-1}},
		{name: "length", rows: []int{values.Len()}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("Take did not panic")
				}
			}()
			values.Take(test.rows)
		})
	}
}

func TestTakeNullable(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	rows := FromOptionals([]Optional[int]{Some(2), None[int](), Some(1), Some(0)})
	taken := source.TakeNullable(rows)
	want := []Optional[int]{Some(30), None[int](), None[int](), Some(10)}
	if got := taken.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("TakeNullable() = %+v, want %+v", got, want)
	}
	if !taken.Nullable() {
		t.Fatal("TakeNullable() returned a non-null Series")
	}

	dense := New([]int{10, 20}).TakeNullable(New([]int{1, 0}))
	if got := dense.Optionals(); !slices.Equal(got, []Optional[int]{Some(20), Some(10)}) || !dense.Nullable() {
		t.Fatalf("dense TakeNullable() = %+v, nullable %t", got, dense.Nullable())
	}
}

func TestTakeNullablePanicsOnInvalidPresentIndex(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("TakeNullable did not panic")
		}
	}()
	New([]int{10}).TakeNullable(FromOptionals([]Optional[int]{Some(-1), None[int]()}))
}

func BenchmarkTake(b *testing.B) {
	const length = 1 << 16
	rows := make([]int, length)
	for i := range rows {
		rows[i] = i
	}
	shuffled := slices.Clone(rows)
	rand.New(rand.NewPCG(1, 2)).Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	values := Repeat(1, length)
	benchmarks := []struct {
		name string
		rows []int
	}{
		{name: "ordered", rows: rows},
		{name: "shuffled", rows: shuffled},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			var result Series[int]
			for b.Loop() {
				result = values.Take(benchmark.rows)
			}
			runtime.KeepAlive(result)
		})
	}
}

func BenchmarkTakeNullable(b *testing.B) {
	const length = 1 << 16
	values := Repeat(1, length)
	rows := make([]Optional[int], length)
	for i := range rows {
		if i%4 != 0 {
			rows[i] = Some(i)
		}
	}
	indexes := FromOptionals(rows)
	b.ReportAllocs()
	var result Series[int]
	for b.Loop() {
		result = values.TakeNullable(indexes)
	}
	runtime.KeepAlive(result)
}

func TestHeadAndTail(t *testing.T) {
	values := New([]int{10, 20, 30, 40})
	tests := []struct {
		name string
		got  Series[int]
		want []int
	}{
		{name: "empty head", got: values.Head(0), want: nil},
		{name: "head", got: values.Head(2), want: []int{10, 20}},
		{name: "large head", got: values.Head(10), want: []int{10, 20, 30, 40}},
		{name: "empty tail", got: values.Tail(0), want: nil},
		{name: "tail", got: values.Tail(2), want: []int{30, 40}},
		{name: "large tail", got: values.Tail(10), want: []int{10, 20, 30, 40}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.got.Values(); !slices.Equal(got, test.want) {
				t.Fatalf("values = %v, want %v", got, test.want)
			}
		})
	}

	nullable := FromOptionals([]Optional[int]{Some(10), None[int]()})
	if !nullable.Head(0).Nullable() || !nullable.Tail(0).Nullable() {
		t.Fatal("Head or Tail removed nullable schema from an empty result")
	}
}

func TestHeadAndTailPanicOnNegativeCount(t *testing.T) {
	values := New([]int{10, 20})
	tests := []struct {
		name string
		call func()
	}{
		{name: "Head", call: func() { values.Head(-1) }},
		{name: "Tail", call: func() { values.Tail(-1) }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("call did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestSlice(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30), Some(40)})
	sliced := source.Slice(1, 3)
	if got := sliced.Optionals(); !slices.Equal(got, []Optional[int]{None[int](), Some(30)}) {
		t.Fatalf("Slice(1, 3) = %+v, want null, 30", got)
	}
	if !sliced.Nullable() {
		t.Fatal("Slice() removed nullable schema")
	}
	if got := slices.Collect(sliced.IsNull().Rows()); !slices.Equal(got, []int{0}) {
		t.Fatalf("Slice().IsNull() rows = %v, want [0]", got)
	}
	if got := slices.Collect(sliced.IsNotNull().Rows()); !slices.Equal(got, []int{1}) {
		t.Fatalf("Slice().IsNotNull() rows = %v, want [1]", got)
	}
	if &sliced.values[0] != &source.values[1] {
		t.Fatal("Slice() did not share value storage")
	}
	source.validity.Set(2, false)
	if sliced.IsValid(1) {
		t.Fatal("Slice() did not share validity storage")
	}
	if cap(sliced.values) != sliced.Len() {
		t.Fatal("Slice() values can extend into adjacent source rows")
	}

	empty := source.Slice(2, 2)
	if empty.Len() != 0 || !empty.Nullable() {
		t.Fatalf("empty Slice() = {Len:%d Nullable:%t}", empty.Len(), empty.Nullable())
	}
}

func TestSlicePanicsOnInvalidBounds(t *testing.T) {
	values := New([]int{10, 20, 30})
	tests := []struct {
		name       string
		start, end int
	}{
		{name: "negative start", start: -1, end: 1},
		{name: "reversed", start: 2, end: 1},
		{name: "past end", start: 0, end: 4},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("Slice did not panic")
				}
			}()
			values.Slice(test.start, test.end)
		})
	}
}

func BenchmarkSliceViews(b *testing.B) {
	values := Repeat(1, 1<<16)
	benchmarks := []struct {
		name string
		view func() Series[int]
	}{
		{name: "Slice", view: func() Series[int] { return values.Slice(1, values.Len()-1) }},
		{name: "Head", view: func() Series[int] { return values.Head(values.Len() / 2) }},
		{name: "Tail", view: func() Series[int] { return values.Tail(values.Len() / 2) }},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			var result Series[int]
			for b.Loop() {
				result = benchmark.view()
			}
			runtime.KeepAlive(result)
		})
	}
}

func TestNullMasks(t *testing.T) {
	tests := []struct {
		name   string
		values Series[int]
		nulls  []bool
	}{
		{name: "empty non-null", values: Series[int]{}, nulls: nil},
		{name: "empty nullable", values: FromOptionals[int](nil), nulls: nil},
		{name: "non-null", values: New([]int{10, 20, 30}), nulls: []bool{false, false, false}},
		{name: "all present nullable", values: FromOptionals([]Optional[int]{Some(10), Some(20)}), nulls: []bool{false, false}},
		{name: "mixed", values: FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)}), nulls: []bool{false, true, false}},
		{name: "all null", values: FromOptionals([]Optional[int]{None[int](), None[int]()}), nulls: []bool{true, true}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nulls := test.values.IsNull()
			present := test.values.IsNotNull()
			if nulls.Len() != test.values.Len() || present.Len() != test.values.Len() {
				t.Fatalf("mask lengths = (%d, %d), want %d", nulls.Len(), present.Len(), test.values.Len())
			}
			for i, wantNull := range test.nulls {
				if got := nulls.At(i); got != wantNull {
					t.Errorf("IsNull().At(%d) = %t, want %t", i, got, wantNull)
				}
				wantPresent := !wantNull
				if got := present.At(i); got != wantPresent {
					t.Errorf("IsNotNull().At(%d) = %t, want %t", i, got, wantPresent)
				}
			}
		})
	}
}

func BenchmarkNullMasks(b *testing.B) {
	const length = 1 << 16
	values := make([]int, length)
	allPresent := make([]bool, length)
	partiallyNull := make([]bool, length)
	for i := range values {
		allPresent[i] = true
		partiallyNull[i] = i%4 != 0
	}
	allPresentValues, err := NewNullable(values, allPresent)
	if err != nil {
		b.Fatal(err)
	}
	partiallyNullValues, err := NewNullable(values, partiallyNull)
	if err != nil {
		b.Fatal(err)
	}
	allNullValues, err := NewNullable(values, make([]bool, length))
	if err != nil {
		b.Fatal(err)
	}
	inputs := []struct {
		name   string
		values Series[int]
	}{
		{name: "non-null", values: New(values)},
		{name: "nullable/all-present", values: allPresentValues},
		{name: "nullable/25%-null", values: partiallyNullValues},
		{name: "nullable/all-null", values: allNullValues},
	}
	operations := []struct {
		name string
		mask func(Series[int]) mask.Mask
	}{
		{name: "IsNull", mask: func(values Series[int]) mask.Mask { return values.IsNull() }},
		{name: "IsNotNull", mask: func(values Series[int]) mask.Mask { return values.IsNotNull() }},
	}

	for _, operation := range operations {
		b.Run(operation.name, func(b *testing.B) {
			for _, input := range inputs {
				b.Run(input.name, func(b *testing.B) {
					b.ReportAllocs()
					var count int
					for b.Loop() {
						count = operation.mask(input.values).Count()
					}
					runtime.KeepAlive(count)
				})
			}
		})
	}
}

func TestFillNull(t *testing.T) {
	tests := []struct {
		name   string
		values Series[int]
		want   []int
		share  bool
	}{
		{name: "empty non-null", values: Series[int]{}, want: nil, share: true},
		{name: "empty nullable", values: FromOptionals[int](nil), want: nil, share: true},
		{name: "non-null", values: New([]int{10, 20}), want: []int{10, 20}, share: true},
		{name: "all present nullable", values: FromOptionals([]Optional[int]{Some(10), Some(20)}), want: []int{10, 20}, share: true},
		{name: "mixed", values: FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)}), want: []int{10, 99, 30}},
		{name: "all null", values: FromOptionals([]Optional[int]{None[int](), None[int]()}), want: []int{99, 99}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			before := test.values.Optionals()
			filled := test.values.FillNull(99)
			if filled.Nullable() {
				t.Fatal("FillNull() returned a nullable Series")
			}
			if got := filled.Values(); !slices.Equal(got, test.want) {
				t.Fatalf("FillNull(99) = %v, want %v", got, test.want)
			}
			if test.share && test.values.Len() != 0 && &filled.values[0] != &test.values.values[0] {
				t.Fatal("FillNull() copied values when no rows changed")
			}
			if got := test.values.Optionals(); !slices.Equal(got, before) {
				t.Fatalf("FillNull() changed source from %+v to %+v", before, got)
			}
		})
	}
}

func TestDropNull(t *testing.T) {
	tests := []struct {
		name   string
		values Series[int]
		want   []int
		share  bool
	}{
		{name: "empty non-null", values: Series[int]{}, want: nil, share: true},
		{name: "empty nullable", values: FromOptionals[int](nil), want: nil, share: true},
		{name: "non-null", values: New([]int{10, 20}), want: []int{10, 20}, share: true},
		{name: "all present nullable", values: FromOptionals([]Optional[int]{Some(10), Some(20)}), want: []int{10, 20}, share: true},
		{name: "mixed", values: FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)}), want: []int{10, 30}},
		{name: "all null", values: FromOptionals([]Optional[int]{None[int](), None[int]()}), want: nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			before := test.values.Optionals()
			dropped := test.values.DropNull()
			if dropped.Nullable() {
				t.Fatal("DropNull() returned a nullable Series")
			}
			if got := dropped.Values(); !slices.Equal(got, test.want) {
				t.Fatalf("DropNull() = %v, want %v", got, test.want)
			}
			if test.share && test.values.Len() != 0 && &dropped.values[0] != &test.values.values[0] {
				t.Fatal("DropNull() copied values when no rows changed")
			}
			if got := test.values.Optionals(); !slices.Equal(got, before) {
				t.Fatalf("DropNull() changed source from %+v to %+v", before, got)
			}
		})
	}
}

func BenchmarkNullTransforms(b *testing.B) {
	const length = 1 << 16
	values := make([]int, length)
	allPresent := make([]bool, length)
	partiallyNull := make([]bool, length)
	for i := range values {
		values[i] = i
		allPresent[i] = true
		partiallyNull[i] = i%4 != 0
	}
	allPresentValues, err := NewNullable(values, allPresent)
	if err != nil {
		b.Fatal(err)
	}
	partiallyNullValues, err := NewNullable(values, partiallyNull)
	if err != nil {
		b.Fatal(err)
	}
	allNullValues, err := NewNullable(values, make([]bool, length))
	if err != nil {
		b.Fatal(err)
	}
	inputs := []struct {
		name   string
		values Series[int]
	}{
		{name: "non-null", values: New(values)},
		{name: "nullable/all-present", values: allPresentValues},
		{name: "nullable/25%-null", values: partiallyNullValues},
		{name: "nullable/all-null", values: allNullValues},
	}
	operations := []struct {
		name      string
		transform func(Series[int]) Series[int]
	}{
		{name: "FillNull", transform: func(values Series[int]) Series[int] { return values.FillNull(0) }},
		{name: "DropNull", transform: func(values Series[int]) Series[int] { return values.DropNull() }},
	}

	for _, operation := range operations {
		b.Run(operation.name, func(b *testing.B) {
			for _, input := range inputs {
				b.Run(input.name, func(b *testing.B) {
					b.ReportAllocs()
					var result Series[int]
					for b.Loop() {
						result = operation.transform(input.values)
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}

func TestEqualFunc(t *testing.T) {
	hiddenLeft, err := NewNullable([]int{10, 20}, []bool{true, false})
	if err != nil {
		t.Fatal(err)
	}
	hiddenRight, err := NewNullable([]int{10, 99}, []bool{true, false})
	if err != nil {
		t.Fatal(err)
	}
	allPresent, err := NewNullable([]int{10, 20}, []bool{true, true})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name  string
		left  Series[int]
		right Series[int]
		want  bool
	}{
		{name: "equal", left: New([]int{10, 20}), right: New([]int{10, 20}), want: true},
		{name: "different length", left: New([]int{10}), right: New([]int{10, 20})},
		{name: "different value", left: New([]int{10, 20}), right: New([]int{10, 30})},
		{name: "different validity", left: hiddenLeft, right: allPresent},
		{name: "hidden values ignored", left: hiddenLeft, right: hiddenRight, want: true},
		{name: "nullable schema ignored", left: New([]int{10, 20}), right: allPresent, want: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.left.EqualFunc(test.right, func(left, right int) bool { return left == right }); got != test.want {
				t.Fatalf("EqualFunc() = %t, want %t", got, test.want)
			}
		})
	}

	left := New([][]int{{1, 2}, {3}})
	right := New([][]int{{1, 2}, {3}})
	if !left.EqualFunc(right, slices.Equal) {
		t.Fatal("EqualFunc() could not compare slice values")
	}
}

func TestEqualFuncPanicsForNilFunction(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("EqualFunc(nil) did not panic")
		}
	}()
	Series[int]{}.EqualFunc(Series[int]{}, nil)
}

func TestEqual(t *testing.T) {
	if !Equal(New([]string{"go"}), New([]string{"go"})) {
		t.Fatal("Equal() = false for equal Series")
	}
	if Equal(New([]float64{math.NaN()}), New([]float64{math.NaN()})) {
		t.Fatal("Equal() matched NaN values")
	}
}

func TestConcat(t *testing.T) {
	left := New([]int{10, 20})
	middle := FromOptionals([]Optional[int]{None[int](), Some(40)})
	right := New([]int{50})
	result := left.Concat(middle, right)
	want := []Optional[int]{Some(10), Some(20), None[int](), Some(40), Some(50)}
	if got := result.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("Concat() = %+v, want %+v", got, want)
	}
	if !result.Nullable() {
		t.Fatal("Concat() did not widen nullable schema")
	}

	dense := left.Concat(right)
	if dense.Nullable() || !slices.Equal(dense.Values(), []int{10, 20, 50}) {
		t.Fatalf("dense Concat() = {Values:%v Nullable:%t}", dense.Values(), dense.Nullable())
	}

	widened := left.Concat(FromOptionals[int](nil))
	if !widened.Nullable() || !slices.Equal(widened.Values(), []int{10, 20}) {
		t.Fatalf("Concat(nullable empty) = {Values:%v Nullable:%t}", widened.Values(), widened.Nullable())
	}

	unchanged := left.Concat()
	if &unchanged.values[0] != &left.values[0] {
		t.Fatal("Concat() copied its receiver with no inputs")
	}
}

func BenchmarkConcat(b *testing.B) {
	parts := make([]Series[int], 8)
	for i := range parts {
		parts[i] = Repeat(i, 1<<10)
	}
	first := parts[0]
	rest := parts[1:]
	b.ReportAllocs()
	var result Series[int]
	for b.Loop() {
		result = first.Concat(rest...)
	}
	runtime.KeepAlive(result)
}

func TestMap(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	calls := 0
	mapped := source.Map(func(value int) string {
		calls++
		return string(rune('a' + value/10 - 1))
	})
	want := []Optional[string]{Some("a"), None[string](), Some("c")}
	if got := mapped.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("Map() = %+v, want %+v", got, want)
	}
	if calls != 2 || !mapped.Nullable() {
		t.Fatalf("Map() = {Calls:%d Nullable:%t}", calls, mapped.Nullable())
	}

	dense := New([]int{1, 2}).Map(func(value int) int { return value * 2 })
	if dense.Nullable() || !slices.Equal(dense.Values(), []int{2, 4}) {
		t.Fatalf("dense Map() = {Values:%v Nullable:%t}", dense.Values(), dense.Nullable())
	}
}

func TestMapCells(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	calls := 0
	mapped := source.MapCells(func(cell Optional[int]) Optional[string] {
		calls++
		if !cell.Valid {
			return Some("missing")
		}
		if cell.Value == 30 {
			return None[string]()
		}
		return Some("present")
	})
	want := []Optional[string]{Some("present"), Some("missing"), None[string]()}
	if got := mapped.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("MapCells() = %+v, want %+v", got, want)
	}
	if calls != source.Len() || !mapped.Nullable() {
		t.Fatalf("MapCells() = {Calls:%d Nullable:%t}", calls, mapped.Nullable())
	}
}

func TestMapOptional(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	calls := 0
	mapped := source.MapOptional(func(value int) (int, bool) {
		calls++
		return value * 2, value != 30
	})
	want := []Optional[int]{Some(20), None[int](), None[int]()}
	if got := mapped.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("MapOptional() = %+v, want %+v", got, want)
	}
	if calls != 2 || !mapped.Nullable() {
		t.Fatalf("MapOptional() = {Calls:%d Nullable:%t}", calls, mapped.Nullable())
	}
}

func TestTryMap(t *testing.T) {
	stop := errors.New("stop")
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	_, err := source.TryMap(func(value int) (int, error) {
		if value == 30 {
			return 0, stop
		}
		return value * 2, nil
	})
	if !errors.Is(err, stop) || err.Error() != "series: TryMap: row 2: stop" {
		t.Fatalf("TryMap() error = %v", err)
	}

	mapped, err := source.TryMap(func(value int) (int, error) { return value * 2, nil })
	if err != nil {
		t.Fatal(err)
	}
	if got := mapped.Optionals(); !slices.Equal(got, []Optional[int]{Some(20), None[int](), Some(60)}) {
		t.Fatalf("TryMap() = %+v", got)
	}
}

func TestTryMapCells(t *testing.T) {
	stop := errors.New("stop")
	source := FromOptionals([]Optional[int]{Some(10), None[int]()})
	_, err := source.TryMapCells(func(cell Optional[int]) (Optional[int], error) {
		if !cell.Valid {
			return None[int](), stop
		}
		return Some(cell.Value), nil
	})
	if !errors.Is(err, stop) || err.Error() != "series: TryMapCells: row 1: stop" {
		t.Fatalf("TryMapCells() error = %v", err)
	}
}

func BenchmarkMap(b *testing.B) {
	values := Repeat(1, 1<<16)
	b.ReportAllocs()
	var result Series[int]
	for b.Loop() {
		result = values.Map(func(value int) int { return value + 1 })
	}
	runtime.KeepAlive(result)
}

func TestMap2(t *testing.T) {
	left := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	right := FromOptionals([]Optional[int]{Some(1), Some(2), None[int]()})
	calls := 0
	mapped := left.Map2(right, func(left, right int) int {
		calls++
		return left + right
	})
	if got := mapped.Optionals(); !slices.Equal(got, []Optional[int]{Some(11), None[int](), None[int]()}) {
		t.Fatalf("Map2() = %+v", got)
	}
	if calls != 1 || !mapped.Nullable() {
		t.Fatalf("Map2() = {Calls:%d Nullable:%t}", calls, mapped.Nullable())
	}

	dense := New([]int{1, 2}).Map2(New([]int{3, 4}), func(left, right int) int { return left + right })
	if dense.Nullable() || !slices.Equal(dense.Values(), []int{4, 6}) {
		t.Fatalf("dense Map2() = {Values:%v Nullable:%t}", dense.Values(), dense.Nullable())
	}
}

func TestMap2Cells(t *testing.T) {
	left := FromOptionals([]Optional[int]{Some(10), None[int]()})
	right := FromOptionals([]Optional[int]{None[int](), Some(2)})
	mapped := left.Map2Cells(right, func(left, right Optional[int]) Optional[int] {
		return Some(left.Or(0) + right.Or(0))
	})
	if got := mapped.Optionals(); !slices.Equal(got, []Optional[int]{Some(10), Some(2)}) {
		t.Fatalf("Map2Cells() = %+v", got)
	}
}

func TestTryMap2(t *testing.T) {
	stop := errors.New("stop")
	left := New([]int{1, 2})
	right := New([]int{10, 20})
	_, err := left.TryMap2(right, func(left, right int) (int, error) {
		if left == 2 {
			return 0, stop
		}
		return left + right, nil
	})
	if !errors.Is(err, stop) || err.Error() != "series: TryMap2: row 1: stop" {
		t.Fatalf("TryMap2() error = %v", err)
	}

	_, err = left.TryMap2Cells(right, func(left, right Optional[int]) (Optional[int], error) {
		if left.Value == 2 {
			return None[int](), stop
		}
		return Some(left.Value + right.Value), nil
	})
	if !errors.Is(err, stop) || err.Error() != "series: TryMap2Cells: row 1: stop" {
		t.Fatalf("TryMap2Cells() error = %v", err)
	}
}

func TestMap2LengthMismatchPanics(t *testing.T) {
	left := New([]int{1})
	right := New([]int{1, 2})
	tests := []struct {
		name string
		call func()
	}{
		{name: "Map2", call: func() { left.Map2(right, func(a, b int) int { return a + b }) }},
		{name: "Map2Cells", call: func() { left.Map2Cells(right, func(a, b Optional[int]) Optional[int] { return a }) }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("call did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestTryMap2LengthMismatchReturnsError(t *testing.T) {
	left := New([]int{1})
	right := New([]int{1, 2})
	_, err := left.TryMap2(right, func(a, b int) (int, error) { return a + b, nil })
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("TryMap2() length error = %v, want ErrLengthMismatch", err)
	}
	_, err = left.TryMap2Cells(right, func(a, b Optional[int]) (Optional[int], error) { return a, nil })
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("TryMap2Cells() length error = %v, want ErrLengthMismatch", err)
	}
}

func TestScanAndReduce(t *testing.T) {
	values := FromOptionals([]Optional[int]{Some(1), None[int](), Some(3)})
	scanned := values.Scan(0, func(sum, value int) int { return sum + value })
	if got := scanned.Optionals(); !slices.Equal(got, []Optional[int]{Some(1), None[int](), Some(4)}) {
		t.Fatalf("Scan() = %+v", got)
	}
	if !scanned.Nullable() {
		t.Fatal("Scan() removed nullable schema")
	}
	if got := values.Reduce(0, func(sum, value int) int { return sum + value }); got != 4 {
		t.Fatalf("Reduce() = %d, want 4", got)
	}
}

func TestSortedFunc(t *testing.T) {
	type item struct {
		key int
		id  string
	}
	values, err := NewNullable(
		[]item{{2, "a"}, {0, "null1"}, {1, "b"}, {2, "c"}, {0, "null2"}},
		[]bool{true, false, true, true, false},
	)
	if err != nil {
		t.Fatal(err)
	}
	sorted := values.SortedFunc(func(left, right item) int {
		if left.id == "null1" || left.id == "null2" || right.id == "null1" || right.id == "null2" {
			t.Fatal("comparator received a null physical value")
		}
		return left.key - right.key
	})
	gotValues := sorted.Values()
	gotIDs := make([]string, len(gotValues))
	for i, value := range gotValues {
		gotIDs[i] = value.id
	}
	if !slices.Equal(gotIDs, []string{"b", "a", "c", "null1", "null2"}) {
		t.Fatalf("SortedFunc() order = %v", gotIDs)
	}
	if got := sorted.Validity(); !slices.Equal(got, []bool{true, true, true, false, false}) {
		t.Fatalf("SortedFunc() validity = %v", got)
	}
	if got := values.Values(); got[0].id != "a" {
		t.Fatal("SortedFunc() changed source")
	}
}

func BenchmarkSortedFunc(b *testing.B) {
	values := make([]int, 1<<14)
	for i := range values {
		values[i] = len(values) - i
	}
	source := New(values)
	b.ReportAllocs()
	var result Series[int]
	for b.Loop() {
		result = source.SortedFunc(func(left, right int) int { return left - right })
	}
	runtime.KeepAlive(result)
}
