package dataframe

import (
	"fmt"
	"hash/maphash"

	"github.com/joeychilson/dataframe/internal/hashmap"
	"github.com/joeychilson/dataframe/series"
)

// JoinKey is an opaque typed join condition. Construct one with On, OnBy,
// Using, UsingBy, UsingColumns, or UsingColumnsBy.
//
// On and OnBy join positional Series and retain all columns from both frames.
// Using variants resolve stored columns by name and emit one coalesced key.
// Null keys never match. Comparable joins use Go ==, so NaN does not match
// itself; By variants use the supplied Hasher's equivalence relation.
type JoinKey[K any] struct {
	left        series.Series[K]
	right       series.Series[K]
	newLookup   func(int) joinLookup[K]
	leftName    string
	rightName   string
	outputName  string
	usingStored bool
}

// On joins positional comparable keys using ==. It retains every original
// column from both frames; name collisions return ErrColumnConflict.
func On[K comparable](left, right series.Series[K]) JoinKey[K] {
	return JoinKey[K]{
		left:  left,
		right: right,
		newLookup: func(capacity int) joinLookup[K] {
			return make(comparableJoinLookup[K], capacity)
		},
	}
}

// OnBy is On with a custom hash and equivalence relation. It supports
// non-comparable keys.
func OnBy[K any](left, right series.Series[K], hasher maphash.Hasher[K]) JoinKey[K] {
	var newLookup func(int) joinLookup[K]
	if hasher != nil {
		newLookup = func(int) joinLookup[K] {
			return hashmap.New[K, joinMatch](hasher)
		}
	}
	return JoinKey[K]{left: left, right: right, newLookup: newLookup}
}

// Using joins same-named stored columns using == and emits one coalesced column
// under name. K must be supplied explicitly because it cannot be inferred from
// a column name.
func Using[K comparable](name string) JoinKey[K] {
	return UsingColumns[K](name, name, name)
}

// UsingBy is Using with a custom hash and equivalence relation. It supports
// non-comparable stored key columns. K must be supplied explicitly.
func UsingBy[K any](name string, hasher maphash.Hasher[K]) JoinKey[K] {
	return UsingColumnsBy(name, name, name, hasher)
}

// UsingColumns joins differently named stored columns using == and emits one
// coalesced key under outputName. K must be supplied explicitly.
func UsingColumns[K comparable](outputName, leftName, rightName string) JoinKey[K] {
	return JoinKey[K]{
		newLookup: func(capacity int) joinLookup[K] {
			return make(comparableJoinLookup[K], capacity)
		},
		leftName:    leftName,
		rightName:   rightName,
		outputName:  outputName,
		usingStored: true,
	}
}

// UsingColumnsBy is UsingColumns with a custom hash and equivalence relation.
// It supports non-comparable stored key columns. K must be supplied explicitly.
func UsingColumnsBy[K any](outputName, leftName, rightName string, hasher maphash.Hasher[K]) JoinKey[K] {
	var newLookup func(int) joinLookup[K]
	if hasher != nil {
		newLookup = func(int) joinLookup[K] {
			return hashmap.New[K, joinMatch](hasher)
		}
	}
	return JoinKey[K]{
		newLookup:   newLookup,
		leftName:    leftName,
		rightName:   rightName,
		outputName:  outputName,
		usingStored: true,
	}
}

// InnerJoin returns matching row pairs. Rows are left-major, and matches for a
// left row retain right-row order. Positional key length, dynamic schema, and
// output name conflicts return errors. On variants produce the complete left
// schema followed by the complete right schema. Using variants place the
// coalesced key at the left key's position, omit the right key, and otherwise
// retain left-then-right schema order.
func (f Frame) InnerJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	resolved, leftKeyIndex, rightKeyIndex, err := resolveJoinKey(f, right, key)
	if err != nil {
		return Frame{}, err
	}
	if err := validateJoinSchema(f, right, resolved, leftKeyIndex, rightKeyIndex); err != nil {
		return Frame{}, err
	}
	lookup := resolved.newLookup(resolved.right.Len())
	leftRows, rightRows := matchingRows(resolved.left, resolved.right, lookup, false)
	return buildJoinFrame(f, right, resolved, leftKeyIndex, rightKeyIndex, leftRows, rightRows, innerJoin), nil
}

// LeftJoin returns every left row once per match, or once with null right
// values when unmatched. Rows are left-major. Invalid keys and output schemas
// return errors. Output schema follows InnerJoin; right-side output fields are
// nullable.
func (f Frame) LeftJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	resolved, leftKeyIndex, rightKeyIndex, err := resolveJoinKey(f, right, key)
	if err != nil {
		return Frame{}, err
	}
	if err := validateJoinSchema(f, right, resolved, leftKeyIndex, rightKeyIndex); err != nil {
		return Frame{}, err
	}
	lookup := resolved.newLookup(resolved.right.Len())
	leftRows, rightRows := matchingRows(resolved.left, resolved.right, lookup, true)
	return buildJoinFrame(f, right, resolved, leftKeyIndex, rightKeyIndex, leftRows, rightRows, leftJoin), nil
}

// RightJoin returns every right row once per match, or once with null left
// values when unmatched. Rows are right-major. Invalid keys and output schemas
// return errors. Output schema remains left then right; left-side output fields
// are nullable. A coalesced key takes its value from the right key for an
// unmatched right row.
func (f Frame) RightJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	resolved, leftKeyIndex, rightKeyIndex, err := resolveJoinKey(f, right, key)
	if err != nil {
		return Frame{}, err
	}
	if err := validateJoinSchema(f, right, resolved, leftKeyIndex, rightKeyIndex); err != nil {
		return Frame{}, err
	}
	lookup := resolved.newLookup(resolved.left.Len())
	rightRows, leftRows := matchingRows(resolved.right, resolved.left, lookup, true)
	return buildJoinFrame(f, right, resolved, leftKeyIndex, rightKeyIndex, leftRows, rightRows, rightJoin), nil
}

// FullJoin scans left rows in order, emitting their matches or one unmatched
// row, then emits unmatched right rows in right-row order. Invalid keys and
// output schemas return errors. Output schema follows InnerJoin; non-key fields
// from both sides are nullable. A coalesced key takes the present key from
// either side.
func (f Frame) FullJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	resolved, leftKeyIndex, rightKeyIndex, err := resolveJoinKey(f, right, key)
	if err != nil {
		return Frame{}, err
	}
	if err := validateJoinSchema(f, right, resolved, leftKeyIndex, rightKeyIndex); err != nil {
		return Frame{}, err
	}
	lookup := resolved.newLookup(resolved.right.Len())
	leftRows, rightRows := fullMatchingRows(resolved.left, resolved.right, lookup)
	return buildJoinFrame(f, right, resolved, leftKeyIndex, rightKeyIndex, leftRows, rightRows, fullJoin), nil
}

// SemiJoin keeps left rows having at least one right match. It emits no right
// columns, does not multiply left rows, and returns the left schema unchanged.
func (f Frame) SemiJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	resolved, _, _, err := resolveJoinKey(f, right, key)
	if err != nil {
		return Frame{}, err
	}
	lookup := resolved.newLookup(resolved.right.Len())
	rows := matchedLeftRows(resolved.left, resolved.right, lookup, true)
	return f.Take(rows), nil
}

// AntiJoin keeps left rows having no right match and returns the left schema
// unchanged.
func (f Frame) AntiJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	resolved, _, _, err := resolveJoinKey(f, right, key)
	if err != nil {
		return Frame{}, err
	}
	lookup := resolved.newLookup(resolved.right.Len())
	rows := matchedLeftRows(resolved.left, resolved.right, lookup, false)
	return f.Take(rows), nil
}

// CrossJoin returns the Cartesian product. Name collisions return
// ErrColumnConflict.
func (f Frame) CrossJoin(right Frame) (Frame, error) {
	if err := validateColumnNames(f, right, -1, -1, ""); err != nil {
		return Frame{}, err
	}
	maxInt := int(^uint(0) >> 1)
	if right.Len() != 0 && f.Len() > maxInt/right.Len() {
		return Frame{}, fmt.Errorf("%w: cross join row count overflows int", ErrRowCount)
	}
	total := f.Len() * right.Len()
	leftRows := make([]int, 0, total)
	rightRows := make([]int, 0, total)
	for leftRow := 0; leftRow < f.Len(); leftRow++ {
		for rightRow := 0; rightRow < right.Len(); rightRow++ {
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, rightRow)
		}
	}
	columns := make([]ColumnSpec, 0, f.Width()+right.Width())
	for _, column := range f.columns {
		columns = append(columns, column.columnTake(leftRows))
	}
	for _, column := range right.columns {
		columns = append(columns, column.columnTake(rightRows))
	}
	return Frame{columns: columns, rowCount: total}, nil
}

type joinKind uint8

const (
	innerJoin joinKind = iota
	leftJoin
	rightJoin
	fullJoin
)

type joinMatch struct {
	first int
	last  int
}

type joinLookup[K any] interface {
	Get(K) (joinMatch, bool)
	Set(K, joinMatch)
}

type comparableJoinLookup[K comparable] map[K]joinMatch

func (m comparableJoinLookup[K]) Get(key K) (joinMatch, bool) {
	match, found := m[key]
	return match, found
}

func (m comparableJoinLookup[K]) Set(key K, match joinMatch) {
	m[key] = match
}

type joinIndex[K any] struct {
	lookup joinLookup[K]
	next   []int
}

func newJoinIndex[K any](right series.Series[K], lookup joinLookup[K]) joinIndex[K] {
	index := joinIndex[K]{lookup: lookup, next: make([]int, right.Len())}
	for i := range index.next {
		index.next[i] = -1
	}
	for row := 0; row < right.Len(); row++ {
		value, present := right.At(row)
		if !present {
			continue
		}
		matches, found := index.lookup.Get(value)
		if !found {
			index.lookup.Set(value, joinMatch{first: row, last: row})
			continue
		}
		index.next[matches.last] = row
		matches.last = row
		index.lookup.Set(value, matches)
	}
	return index
}

func resolveJoinKey[K any](left, right Frame, key JoinKey[K]) (JoinKey[K], int, int, error) {
	if key.newLookup == nil {
		return key, -1, -1, fmt.Errorf("%w: join key has a nil hasher", ErrUnsupported)
	}
	if !key.usingStored {
		if key.left.Len() != left.Len() || key.right.Len() != right.Len() {
			return key, -1, -1, fmt.Errorf("%w: positional join keys have lengths %d and %d, want %d and %d", ErrRowCount, key.left.Len(), key.right.Len(), left.Len(), right.Len())
		}
		return key, -1, -1, nil
	}
	if key.leftName == "" || key.rightName == "" || key.outputName == "" {
		return key, -1, -1, fmt.Errorf("%w: join key names must not be empty", ErrInvalidName)
	}
	leftKey, err := left.Column[K](key.leftName)
	if err != nil {
		return key, -1, -1, err
	}
	rightKey, err := right.Column[K](key.rightName)
	if err != nil {
		return key, -1, -1, err
	}
	key.left = leftKey
	key.right = rightKey
	return key, left.columnIndex(key.leftName), right.columnIndex(key.rightName), nil
}

func validateJoinSchema[K any](left, right Frame, key JoinKey[K], leftKeyIndex, rightKeyIndex int) error {
	if key.usingStored {
		return validateColumnNames(left, right, leftKeyIndex, rightKeyIndex, key.outputName)
	}
	return validateColumnNames(left, right, -1, -1, "")
}

func validateColumnNames(left, right Frame, leftKeyIndex, rightKeyIndex int, outputName string) error {
	names := make(map[string]struct{}, left.Width()+right.Width())
	for i, column := range left.columns {
		name := column.columnName()
		if i == leftKeyIndex {
			name = outputName
		}
		if _, exists := names[name]; exists {
			return fmt.Errorf("%w: %q", ErrColumnConflict, name)
		}
		names[name] = struct{}{}
	}
	for i, column := range right.columns {
		if i == rightKeyIndex {
			continue
		}
		name := column.columnName()
		if _, exists := names[name]; exists {
			return fmt.Errorf("%w: %q", ErrColumnConflict, name)
		}
		names[name] = struct{}{}
	}
	return nil
}

func matchingRows[K any](left, right series.Series[K], lookup joinLookup[K], includeUnmatched bool) ([]int, []int) {
	index := newJoinIndex(right, lookup)
	leftRows := make([]int, 0, left.Len())
	rightRows := make([]int, 0, left.Len())
	for leftRow := 0; leftRow < left.Len(); leftRow++ {
		value, present := left.At(leftRow)
		if !present {
			if includeUnmatched {
				leftRows = append(leftRows, leftRow)
				rightRows = append(rightRows, -1)
			}
			continue
		}
		matches, found := index.lookup.Get(value)
		if !found {
			if includeUnmatched {
				leftRows = append(leftRows, leftRow)
				rightRows = append(rightRows, -1)
			}
			continue
		}
		for rightRow := matches.first; rightRow >= 0; rightRow = index.next[rightRow] {
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, rightRow)
		}
	}
	return leftRows, rightRows
}

func fullMatchingRows[K any](left, right series.Series[K], lookup joinLookup[K]) ([]int, []int) {
	index := newJoinIndex(right, lookup)
	leftRows := make([]int, 0, left.Len()+len(index.next))
	rightRows := make([]int, 0, left.Len()+len(index.next))
	matchedRight := make([]bool, len(index.next))
	for leftRow := 0; leftRow < left.Len(); leftRow++ {
		value, present := left.At(leftRow)
		if !present {
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, -1)
			continue
		}
		matches, found := index.lookup.Get(value)
		if !found {
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, -1)
			continue
		}
		for rightRow := matches.first; rightRow >= 0; rightRow = index.next[rightRow] {
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, rightRow)
			matchedRight[rightRow] = true
		}
	}
	for rightRow, matched := range matchedRight {
		if !matched {
			leftRows = append(leftRows, -1)
			rightRows = append(rightRows, rightRow)
		}
	}
	return leftRows, rightRows
}

func matchedLeftRows[K any](left, right series.Series[K], lookup joinLookup[K], wantMatch bool) []int {
	index := newJoinIndex(right, lookup)
	rows := make([]int, 0, left.Len())
	for row := 0; row < left.Len(); row++ {
		value, present := left.At(row)
		if !present {
			if !wantMatch {
				rows = append(rows, row)
			}
			continue
		}
		_, found := index.lookup.Get(value)
		if found == wantMatch {
			rows = append(rows, row)
		}
	}
	return rows
}

func buildJoinFrame[K any](left, right Frame, key JoinKey[K], leftKeyIndex, rightKeyIndex int, leftRows, rightRows []int, kind joinKind) Frame {
	var nullableLeftRows series.Series[int]
	var nullableRightRows series.Series[int]
	switch kind {
	case leftJoin:
		nullableRightRows = joinRowIndexes(rightRows)
	case rightJoin:
		nullableLeftRows = joinRowIndexes(leftRows)
	case fullJoin:
		nullableLeftRows = joinRowIndexes(leftRows)
		nullableRightRows = joinRowIndexes(rightRows)
	}

	columns := make([]ColumnSpec, 0, left.Width()+right.Width())
	for i, column := range left.columns {
		if i == leftKeyIndex {
			joined := joinedKey(key, leftRows, rightRows, nullableLeftRows, nullableRightRows, kind)
			columns = append(columns, ColumnFromSeries(key.outputName, joined))
			continue
		}
		if kind == rightJoin || kind == fullJoin {
			columns = append(columns, column.columnTakeNullable(nullableLeftRows))
		} else {
			columns = append(columns, column.columnTake(leftRows))
		}
	}
	for i, column := range right.columns {
		if i == rightKeyIndex {
			continue
		}
		if kind == leftJoin || kind == fullJoin {
			columns = append(columns, column.columnTakeNullable(nullableRightRows))
		} else {
			columns = append(columns, column.columnTake(rightRows))
		}
	}
	return Frame{columns: columns, rowCount: len(leftRows)}
}

func joinedKey[K any](key JoinKey[K], leftRows, rightRows []int, nullableLeftRows, nullableRightRows series.Series[int], kind joinKind) series.Series[K] {
	switch kind {
	case innerJoin:
		left := key.left.Take(leftRows)
		if !key.left.Nullable() && !key.right.Nullable() {
			return left
		}
		return coalesceJoinKeys(left, key.right.Take(rightRows))
	case leftJoin:
		return key.left.Take(leftRows)
	case rightJoin:
		return key.right.Take(rightRows)
	case fullJoin:
		result := coalesceJoinKeys(key.left.TakeNullable(nullableLeftRows), key.right.TakeNullable(nullableRightRows))
		if !key.left.Nullable() && !key.right.Nullable() {
			return result.DropNull()
		}
		return result
	default:
		panic("dataframe: invalid join kind")
	}
}

func coalesceJoinKeys[K any](left, right series.Series[K]) series.Series[K] {
	return left.Map2Cells(right, func(left, right series.Optional[K]) series.Optional[K] {
		if left.Valid {
			return left
		}
		return right
	})
}

func joinRowIndexes(rows []int) series.Series[int] {
	values := make([]int, len(rows))
	validity := make([]bool, len(rows))
	for i, row := range rows {
		if row >= 0 {
			values[i] = row
			validity[i] = true
		}
	}
	result, err := series.NewNullable(values, validity)
	if err != nil {
		panic(err)
	}
	return result
}
