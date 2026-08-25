package dataframe

import (
	"fmt"
	"hash/maphash"
	"reflect"

	"github.com/joeychilson/dataframe/internal/hashmap"
	"github.com/joeychilson/dataframe/series"
)

// Distinct keeps the first row for each distinct complete row. It returns
// ErrUnsupported when any participating column cannot be compared using its
// natural Go equality. Nulls compare equal to nulls.
func (f Frame) Distinct() (Frame, error) {
	if f.Len() < 2 {
		return f, nil
	}
	if f.Width() == 0 {
		return f.Slice(0, 1), nil
	}
	if f.Width() == 1 {
		stored := f.columns[0]
		if !stored.typeOf.Comparable() {
			return Frame{}, fmt.Errorf("%w: column %q has type %v", ErrUnsupported, stored.name, stored.typeOf)
		}
		if values, ok := newDistinctColumn(stored); ok {
			return f.Take(values.rows()), nil
		}

		seen := make(map[any]struct{}, f.Len())
		rows := make([]int, 0, f.Len())
		nullSeen := false
		for row := range f.Len() {
			value, present := stored.values.at(row)
			if !present {
				if !nullSeen {
					nullSeen = true
					rows = append(rows, row)
				}
				continue
			}
			if value != nil {
				valueOf := reflect.ValueOf(value)
				if !valueOf.Comparable() {
					return Frame{}, fmt.Errorf("%w: column %q row %d contains incomparable dynamic type %v", ErrUnsupported, stored.name, row, valueOf.Type())
				}
			}
			if _, exists := seen[value]; exists {
				continue
			}
			seen[value] = struct{}{}
			rows = append(rows, row)
		}
		return f.Take(rows), nil
	}
	if rows, ok := distinctBuiltinFrameRows(f.columns, f.Len()); ok {
		return f.Take(rows), nil
	}

	// This fallback handles multi-column frames containing at least one column
	// that is not builtin typedData. ValidN fields distinguish nulls from the
	// corresponding ValueN zero values.
	fields := make([]reflect.StructField, 0, f.Width()*2)
	for i, column := range f.columns {
		if !column.typeOf.Comparable() {
			return Frame{}, fmt.Errorf("%w: column %q has type %v", ErrUnsupported, column.name, column.typeOf)
		}
		fields = append(fields,
			reflect.StructField{Name: fmt.Sprintf("Value%d", i), Type: column.typeOf},
			reflect.StructField{Name: fmt.Sprintf("Valid%d", i), Type: reflect.TypeFor[bool]()},
		)
	}
	keyType := reflect.StructOf(fields)
	key := reflect.New(keyType).Elem()
	seen := make(map[any]struct{}, f.Len())
	rows := make([]int, 0, f.Len())
	for row := range f.Len() {
		for columnIndex, column := range f.columns {
			value, present := column.values.at(row)
			valueField := key.Field(columnIndex * 2)
			key.Field(columnIndex*2 + 1).SetBool(present)
			if !present || value == nil {
				valueField.SetZero()
				continue
			}
			valueOf := reflect.ValueOf(value)
			if !valueOf.Comparable() {
				return Frame{}, fmt.Errorf("%w: column %q row %d contains incomparable dynamic type %v", ErrUnsupported, column.name, row, valueOf.Type())
			}
			valueField.Set(valueOf)
		}
		value := key.Interface()
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		rows = append(rows, row)
	}
	return f.Take(rows), nil
}

// DistinctBy keeps the first row for each distinct comparable positional key.
// Composite comparable structs provide a Go-native multi-column key. It panics
// on length mismatch.
func (f Frame) DistinctBy[K comparable](key series.Series[K]) Frame {
	if key.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: DistinctBy: length mismatch: frame=%d key=%d", f.Len(), key.Len()))
	}
	return f.Take(distinctRows(key))
}

// DistinctByUsing is DistinctBy for non-comparable keys or custom equality. It
// panics on length mismatch or a nil hasher.
func (f Frame) DistinctByUsing[K any](key series.Series[K], hasher maphash.Hasher[K]) Frame {
	if key.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: DistinctByUsing: length mismatch: frame=%d key=%d", f.Len(), key.Len()))
	}
	seen := hashmap.New[K, struct{}](hasher, key.Len())
	nullSeen := false
	rows := make([]int, 0, key.Len())
	for i := range key.Len() {
		value, present := key.At(i)
		if !present {
			if nullSeen {
				continue
			}
			nullSeen = true
			rows = append(rows, i)
			continue
		}
		if _, loaded := seen.LoadOrStore(value, struct{}{}); loaded {
			continue
		}
		rows = append(rows, i)
	}
	return f.Take(rows)
}

type distinctColumn interface {
	rows() []int
	hash(*maphash.Hash, int)
	equal(int, int) bool
}

type typedDistinctColumn[T comparable] struct {
	values series.Series[T]
}

type distinctRowHasher []distinctColumn

func distinctBuiltinFrameRows(columns []column, length int) ([]int, bool) {
	distinctColumns := make([]distinctColumn, len(columns))
	for i, column := range columns {
		var ok bool
		distinctColumns[i], ok = newDistinctColumn(column)
		if !ok {
			return nil, false
		}
	}

	seen := hashmap.New[int, struct{}](distinctRowHasher(distinctColumns), length)
	rows := make([]int, 0, length)
	for row := range length {
		if _, loaded := seen.LoadOrStore(row, struct{}{}); !loaded {
			rows = append(rows, row)
		}
	}
	return rows, true
}

func newDistinctColumn(column column) (distinctColumn, bool) {
	switch values := column.values.(type) {
	case typedData[bool]:
		return typedDistinctColumn[bool](values), true
	case typedData[string]:
		return typedDistinctColumn[string](values), true
	case typedData[int]:
		return typedDistinctColumn[int](values), true
	case typedData[int8]:
		return typedDistinctColumn[int8](values), true
	case typedData[int16]:
		return typedDistinctColumn[int16](values), true
	case typedData[int32]:
		return typedDistinctColumn[int32](values), true
	case typedData[int64]:
		return typedDistinctColumn[int64](values), true
	case typedData[uint]:
		return typedDistinctColumn[uint](values), true
	case typedData[uint8]:
		return typedDistinctColumn[uint8](values), true
	case typedData[uint16]:
		return typedDistinctColumn[uint16](values), true
	case typedData[uint32]:
		return typedDistinctColumn[uint32](values), true
	case typedData[uint64]:
		return typedDistinctColumn[uint64](values), true
	case typedData[uintptr]:
		return typedDistinctColumn[uintptr](values), true
	case typedData[float32]:
		return typedDistinctColumn[float32](values), true
	case typedData[float64]:
		return typedDistinctColumn[float64](values), true
	case typedData[complex64]:
		return typedDistinctColumn[complex64](values), true
	case typedData[complex128]:
		return typedDistinctColumn[complex128](values), true
	default:
		return nil, false
	}
}

// Hash writes row's validity and present column values into hash.
func (h distinctRowHasher) Hash(hash *maphash.Hash, row int) {
	for _, column := range h {
		column.hash(hash, row)
	}
}

// Equal reports whether two rows have matching validity and present values.
func (h distinctRowHasher) Equal(left, right int) bool {
	for _, column := range h {
		if !column.equal(left, right) {
			return false
		}
	}
	return true
}

func (c typedDistinctColumn[T]) rows() []int {
	return distinctRows(c.values)
}

func (c typedDistinctColumn[T]) hash(hash *maphash.Hash, row int) {
	value, present := c.values.At(row)
	if !present {
		hash.WriteByte(0)
		return
	}
	hash.WriteByte(1)
	maphash.WriteComparable(hash, value)
	if value != value {
		maphash.WriteComparable(hash, row)
	}
}

func (c typedDistinctColumn[T]) equal(left, right int) bool {
	leftValue, leftPresent := c.values.At(left)
	rightValue, rightPresent := c.values.At(right)
	return leftPresent == rightPresent && (!leftPresent || leftValue == rightValue)
}

func distinctRows[T comparable](values series.Series[T]) []int {
	seen := make(map[T]struct{}, values.Len())
	rows := make([]int, 0, values.Len())
	nullSeen := false
	for row := range values.Len() {
		value, present := values.At(row)
		if !present {
			if !nullSeen {
				nullSeen = true
				rows = append(rows, row)
			}
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		rows = append(rows, row)
	}
	return rows
}
