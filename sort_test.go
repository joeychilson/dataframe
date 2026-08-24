package dataframe

import (
	"cmp"
	"math"
	"runtime"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/series"
)

func TestSortKeyModifiers_ControlNullPlacementAndDirection(t *testing.T) {
	key := series.FromOptionals([]series.Optional[int]{series.Some(2), series.None[int](), series.Some(1)})
	frame, err := New(Column("id", []int{0, 1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		key  SortKey
		want []int
	}{
		{name: "Asc orders values with nulls last", key: Asc(key), want: []int{2, 0, 1}},
		{name: "Desc reverses values with nulls last", key: Desc(key), want: []int{0, 2, 1}},
		{name: "NullsFirst moves nulls before ascending values", key: Asc(key).NullsFirst(), want: []int{1, 2, 0}},
		{name: "Reverse retains explicit null placement", key: Asc(key).NullsFirst().Reverse(), want: []int{1, 0, 2}},
		{name: "the last null placement call wins", key: Asc(key).NullsFirst().NullsLast(), want: []int{2, 0, 1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sorted := frame.SortedBy(test.key)
			ids, _ := sorted.Column[int]("id")
			if got := ids.Values(); !slices.Equal(got, test.want) {
				t.Fatalf("ids = %v, want %v", got, test.want)
			}
		})
	}
}

func TestSortedBy_OrdersRowsStablyBySuccessiveKeys(t *testing.T) {
	frame, err := New(
		Column("id", []int{0, 1, 2, 3, 4}),
		Column("group", []string{"b", "a", "a", "b", "a"}),
		Column("value", []int{2, 2, 1, 1, 2}),
	)
	if err != nil {
		t.Fatal(err)
	}
	groups, _ := frame.Column[string]("group")
	values, _ := frame.Column[int]("value")
	sorted := frame.SortedBy(Asc(groups), Desc(values))
	ids, err := sorted.Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got := ids.Values(); !slices.Equal(got, []int{1, 4, 2, 0, 3}) {
		t.Fatalf("sorted ids = %v", got)
	}

	// Equal keys retain their input order.
	stable := frame.SortedBy(ByFunc(values, func(_, _ int) int { return 0 }))
	if &stable.columns[0] != &frame.columns[0] {
		t.Fatal("SortedBy(already sorted) copied an immutable Frame")
	}
	stableIDs, _ := stable.Column[int]("id")
	if got := stableIDs.Values(); !slices.Equal(got, []int{0, 1, 2, 3, 4}) {
		t.Fatalf("stable ids = %v", got)
	}
}

func TestAsc_OrdersNaNLikeCmpCompare(t *testing.T) {
	values := series.New([]float64{1, math.NaN(), -1, math.NaN()})
	frame, err := New(Column("id", []int{0, 1, 2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	ordered, err := frame.SortedBy(Asc(values)).Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got := ordered.Values(); !slices.Equal(got, []int{1, 3, 2, 0}) {
		t.Fatalf("Asc() ids = %v, want [1 3 2 0]", got)
	}
	fallback, err := frame.SortedBy(ByFunc(values, cmp.Compare[float64])).Column[int]("id")
	if err != nil {
		t.Fatal(err)
	}
	if got := fallback.Values(); !slices.Equal(got, ordered.Values()) {
		t.Fatalf("ByFunc(cmp.Compare) ids = %v, want %v", got, ordered.Values())
	}
}

func TestAsc_PrimitiveTypesMatchComparatorFallback(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name string
		run  func(*testing.T)
	}{
		{name: "int", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(-maxInt-1, -1, 0, 1, maxInt))
		}},
		{name: "int8", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(int8(-128), -1, 0, 1, 127))
		}},
		{name: "int16", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(int16(-32768), -1, 0, 1, 32767))
		}},
		{name: "int32", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(int32(-1<<31), -1, 0, 1, 1<<31-1))
		}},
		{name: "int64", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(int64(-1<<63), -1, 0, 1, 1<<63-1))
		}},
		{name: "uint", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(uint(0), 1, ^uint(0), 1))
		}},
		{name: "uint8", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(uint8(0), 1, 255, 1))
		}},
		{name: "uint16", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(uint16(0), 1, 1<<16-1, 1))
		}},
		{name: "uint32", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(uint32(0), 1, 1<<32-1, 1))
		}},
		{name: "uint64", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(uint64(0), 1, 1<<64-1, 1))
		}},
		{name: "uintptr", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(uintptr(0), 1, ^uintptr(0), 1))
		}},
		{name: "float32", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(
				float32(math.Inf(-1)), -math.MaxFloat32, math.Float32frombits(1<<31), 0,
				math.SmallestNonzeroFloat32, math.MaxFloat32, float32(math.Inf(1)), math.Float32frombits(0x7fc00001),
			))
		}},
		{name: "float64", run: func(t *testing.T) {
			t.Helper()
			assertAscMatchesComparator(t, optionalsWithNull(
				math.Inf(-1), -math.MaxFloat64, math.Copysign(0, -1), 0,
				math.SmallestNonzeroFloat64, math.MaxFloat64, math.Inf(1), math.NaN(),
			))
		}},
	}
	for _, test := range tests {
		t.Run(test.name, test.run)
	}
}

func TestAsc_DefinedNumericTypeMatchesComparatorFallback(t *testing.T) {
	type rank int
	assertAscMatchesComparator(t, optionalsWithNull(rank(2), -1, 2, 0))
}

func TestSortedByPanicsOnInvalidKey(t *testing.T) {
	frame, err := New(Column("id", []int{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	assertPanic := func(call func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatal("function did not panic")
			}
		}()
		call()
	}
	assertPanic(func() { frame.SortedBy(Asc(series.New([]int{1}))) })
	assertPanic(func() { frame.SortedBy(SortKey{}) })
	single := frame.Slice(0, 1)
	assertPanic(func() { single.SortedBy(Asc(series.Series[int]{})) })
	assertPanic(func() { single.SortedBy(SortKey{}) })
	assertPanic(func() { ByFunc(series.New([]int{1}), nil) })
}

func FuzzSortedByAgainstStableModel(f *testing.F) {
	f.Add([]byte{4, 1, 7, 3, 2, 9}, false, false, true)
	f.Add([]byte{}, true, true, false)
	f.Fuzz(func(t *testing.T, data []byte, primaryReverse, nullsFirst, secondaryReverse bool) {
		data = data[:min(len(data), 64)]
		primary := make([]series.Optional[int], len(data))
		secondary := make([]int, len(data))
		rows := make([]int, len(data))
		for i, value := range data {
			if value&1 != 0 {
				primary[i] = series.Some(int(int8(value) / 16))
			}
			secondary[i] = int(int8(value ^ 0x5a))
			rows[i] = i
		}
		frame, err := New(Column("row", rows))
		if err != nil {
			t.Fatal(err)
		}
		primaryValues := series.FromOptionals(primary)
		secondaryValues := series.New(secondary)
		primaryKey := Asc(primaryValues)
		fallbackPrimary := ByFunc(primaryValues, cmp.Compare[int])
		if primaryReverse {
			primaryKey = primaryKey.Reverse()
			fallbackPrimary = fallbackPrimary.Reverse()
		}
		if nullsFirst {
			primaryKey = primaryKey.NullsFirst()
			fallbackPrimary = fallbackPrimary.NullsFirst()
		}
		secondaryKey := Asc(secondaryValues)
		fallbackSecondary := ByFunc(secondaryValues, cmp.Compare[int])
		if secondaryReverse {
			secondaryKey = secondaryKey.Reverse()
			fallbackSecondary = fallbackSecondary.Reverse()
		}

		want := slices.Clone(rows)
		slices.SortStableFunc(want, func(left, right int) int {
			leftKey := primary[left]
			rightKey := primary[right]
			switch {
			case !leftKey.Valid && !rightKey.Valid:
			case !leftKey.Valid:
				if nullsFirst {
					return -1
				}
				return 1
			case !rightKey.Valid:
				if nullsFirst {
					return 1
				}
				return -1
			default:
				order := cmp.Compare(leftKey.Value, rightKey.Value)
				if primaryReverse {
					order = cmp.Compare(rightKey.Value, leftKey.Value)
				}
				if order != 0 {
					return order
				}
			}
			if secondaryReverse {
				return cmp.Compare(secondary[right], secondary[left])
			}
			return cmp.Compare(secondary[left], secondary[right])
		})

		ordered, err := frame.SortedBy(primaryKey, secondaryKey).Column[int]("row")
		if err != nil {
			t.Fatal(err)
		}
		if got := ordered.Values(); !slices.Equal(got, want) {
			t.Fatalf("SortedBy(Asc) rows = %v, want %v", got, want)
		}
		fallback, err := frame.SortedBy(fallbackPrimary, fallbackSecondary).Column[int]("row")
		if err != nil {
			t.Fatal(err)
		}
		if got := fallback.Values(); !slices.Equal(got, want) {
			t.Fatalf("SortedBy(ByFunc) rows = %v, want %v", got, want)
		}
	})
}

func optionalsWithNull[T any](values ...T) []series.Optional[T] {
	result := make([]series.Optional[T], 0, len(values)+1)
	for i, value := range values {
		result = append(result, series.Some(value))
		if i == 0 {
			result = append(result, series.None[T]())
		}
	}
	return result
}

func assertAscMatchesComparator[T cmp.Ordered](t *testing.T, values []series.Optional[T]) {
	t.Helper()
	rows := make([]int, len(values))
	for i := range rows {
		rows[i] = i
	}
	frame, err := New(Column("row", rows))
	if err != nil {
		t.Fatal(err)
	}
	keyValues := series.FromOptionals(values)
	orderedKeys := []SortKey{
		Asc(keyValues),
		Desc(keyValues),
		Asc(keyValues).NullsFirst(),
		Desc(keyValues).NullsFirst(),
	}
	fallbackKeys := []SortKey{
		ByFunc(keyValues, cmp.Compare[T]),
		ByFunc(keyValues, cmp.Compare[T]).Reverse(),
		ByFunc(keyValues, cmp.Compare[T]).NullsFirst(),
		ByFunc(keyValues, cmp.Compare[T]).Reverse().NullsFirst(),
	}
	for i := range orderedKeys {
		got, columnErr := frame.SortedBy(orderedKeys[i]).Column[int]("row")
		if columnErr != nil {
			t.Fatal(columnErr)
		}
		want, columnErr := frame.SortedBy(fallbackKeys[i]).Column[int]("row")
		if columnErr != nil {
			t.Fatal(columnErr)
		}
		if !slices.Equal(got.Values(), want.Values()) {
			t.Fatalf("sort variant %d rows = %v, want %v", i, got.Values(), want.Values())
		}
	}
}

func BenchmarkSortedBy(b *testing.B) {
	const size = 10_000
	rows := make([]int, size)
	for i := range rows {
		rows[i] = i
	}
	frame, err := New(Column("row", rows))
	if err != nil {
		b.Fatal(err)
	}
	benchmarks := []struct {
		name   string
		values series.Series[int]
	}{
		{name: "sorted", values: series.NewFunc(size, func(i int) int { return i })},
		{name: "reverse", values: series.NewFunc(size, func(i int) int { return size - i })},
		{name: "100-values", values: series.NewFunc(size, func(i int) int { return (i * 31) % 100 })},
		{name: "permuted", values: series.NewFunc(size, func(i int) int { return (i * 7919) % size })},
		{name: "nullable", values: series.NewNullableFunc(size, func(i int) (int, bool) {
			return (i * 7919) % size, i%10 != 0
		})},
	}
	for _, benchmark := range benchmarks {
		keys := []struct {
			name string
			key  SortKey
		}{
			{name: "Asc", key: Asc(benchmark.values)},
			{name: "ByFunc", key: ByFunc(benchmark.values, cmp.Compare[int])},
		}
		for _, key := range keys {
			b.Run(benchmark.name+"/"+key.name, func(b *testing.B) {
				b.ReportAllocs()
				var result Frame
				for b.Loop() {
					result = frame.SortedBy(key.key)
				}
				runtime.KeepAlive(result)
			})
		}
	}
}
