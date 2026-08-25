package dataframe

import (
	"fmt"
	"hash/maphash"
	"math"

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
		newLookup = func(capacity int) joinLookup[K] {
			return hashmap.New[K, int](hasher, capacity)
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
		newLookup = func(capacity int) joinLookup[K] {
			return hashmap.New[K, int](hasher, capacity)
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
// retain left-then-right schema order. Output row-count overflow returns
// ErrRowCount.
func (f Frame) InnerJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, innerJoin)
}

// LeftJoin returns every left row once per match, or once with null right
// values when unmatched. Rows are left-major. Invalid keys and output schemas
// return errors. Output schema follows InnerJoin; right-side output fields are
// nullable. Output row-count overflow returns ErrRowCount.
func (f Frame) LeftJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, leftJoin)
}

// RightJoin returns every right row once per match, or once with null left
// values when unmatched. Rows are right-major. Invalid keys and output schemas
// return errors. Output schema remains left then right; left-side output fields
// are nullable. A coalesced key takes its value from the right key for an
// unmatched right row. Output row-count overflow returns ErrRowCount.
func (f Frame) RightJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, rightJoin)
}

// FullJoin scans left rows in order, emitting their matches or one unmatched
// row, then emits unmatched right rows in right-row order. Invalid keys and
// output schemas return errors. Output schema follows InnerJoin; non-key fields
// from both sides are nullable. A coalesced key takes the present key from
// either side. Output row-count overflow returns ErrRowCount.
func (f Frame) FullJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, fullJoin)
}

// SemiJoin keeps left rows having at least one right match. It emits no right
// columns, does not multiply left rows, and returns the left schema unchanged.
func (f Frame) SemiJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, semiJoin)
}

// AntiJoin keeps left rows having no right match and returns the left schema
// unchanged.
func (f Frame) AntiJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, antiJoin)
}

// CrossJoin returns the Cartesian product. Name collisions return
// ErrColumnConflict; output row-count overflow returns ErrRowCount.
func (f Frame) CrossJoin(right Frame) (Frame, error) {
	if err := validateColumnNames(f, right, joinProjection{leftKeyIndex: -1, rightKeyIndex: -1}); err != nil {
		return Frame{}, err
	}
	if right.Len() != 0 && f.Len() > math.MaxInt/right.Len() {
		return Frame{}, fmt.Errorf("%w: cross join row count overflows int", ErrRowCount)
	}
	total := f.Len() * right.Len()
	if f.Width() == 0 && right.Width() == 0 {
		return Frame{rowCount: total}, nil
	}
	var leftRows, rightRows []int
	if f.Width() > 0 {
		leftRows = crossJoinLeftRows(f.Len(), right.Len())
	}
	if right.Width() > 0 {
		rightRows = crossJoinRightRows(f.Len(), right.Len())
	}
	columns := make([]column, 0, f.Width()+right.Width())
	for _, column := range f.columns {
		column.values = column.values.take(leftRows)
		columns = append(columns, column)
	}
	for _, column := range right.columns {
		column.values = column.values.take(rightRows)
		columns = append(columns, column)
	}
	return Frame{columns: columns, rowCount: total}, nil
}

type joinKind uint8

const (
	innerJoin joinKind = iota
	leftJoin
	rightJoin
	fullJoin
	semiJoin
	antiJoin
)

type joinMatch struct {
	first int
	count int
}

type joinLookup[K any] interface {
	// Get returns the row or negative duplicate-match reference stored for a key.
	Get(K) (int, bool)
	// Set stores a row or negative duplicate-match reference for a key.
	Set(K, int)
}

type comparableJoinLookup[K comparable] map[K]int

// Get returns the encoded match stored for key.
func (m comparableJoinLookup[K]) Get(key K) (int, bool) {
	encoded, found := m[key]
	return encoded, found
}

// Set stores an encoded match for key.
func (m comparableJoinLookup[K]) Set(key K, encoded int) {
	m[key] = encoded
}

type joinProjection struct {
	leftKeyIndex  int
	rightKeyIndex int
	outputName    string
}

// resolvedJoinPlan is the normalized runtime form of a JoinKey. Positional
// joins retain the default -1 projection indexes; stored joins resolve their
// key series and output projection once before matching begins.
type resolvedJoinPlan[K any] struct {
	left       series.Series[K]
	right      series.Series[K]
	newLookup  func(int) joinLookup[K]
	projection joinProjection
}

// joinIndex stores one right row directly as a non-negative lookup value. A
// negative value -(i+1) indexes duplicates, while next[row] links each
// duplicate to the older match. TestJoinIndexAllocatesDuplicateChainsLazily
// pins the lazy allocation this encoding provides.
type joinIndex[K any] struct {
	lookup     joinLookup[K]
	duplicates []joinMatch
	next       []int
}

func newJoinIndex[K any](right series.Series[K], lookup joinLookup[K]) joinIndex[K] {
	index := joinIndex[K]{lookup: lookup}
	for row := right.Len() - 1; row >= 0; row-- {
		value, present := right.At(row)
		if !present {
			continue
		}
		encoded, found := index.lookup.Get(value)
		if !found {
			index.lookup.Set(value, row)
			continue
		}
		if index.next == nil {
			index.next = make([]int, right.Len())
		}
		if encoded >= 0 {
			index.next[row] = encoded
			index.duplicates = append(index.duplicates, joinMatch{first: row, count: 2})
			index.lookup.Set(value, -len(index.duplicates))
			continue
		}
		matches := &index.duplicates[-encoded-1]
		index.next[row] = matches.first
		matches.first = row
		matches.count++
	}
	return index
}

func (i joinIndex[K]) matches(values series.Series[K], row int) (joinMatch, bool) {
	value, present := values.At(row)
	if !present {
		return joinMatch{}, false
	}
	encoded, found := i.lookup.Get(value)
	if !found {
		return joinMatch{}, false
	}
	if encoded >= 0 {
		return joinMatch{first: encoded, count: 1}, true
	}
	return i.duplicates[-encoded-1], true
}

func joinFrames[K any](left, right Frame, key JoinKey[K], kind joinKind) (Frame, error) {
	plan, err := resolveJoinPlan(left, right, key)
	if err != nil {
		return Frame{}, err
	}
	if kind != semiJoin && kind != antiJoin {
		if validationErr := validateColumnNames(left, right, plan.projection); validationErr != nil {
			return Frame{}, validationErr
		}
	}

	var leftRows, rightRows []int
	switch kind {
	case innerJoin, leftJoin:
		lookup := plan.newLookup(plan.right.Len())
		leftRows, rightRows, err = pairedMatchingRows(plan.left, plan.right, lookup, kind == leftJoin)
	case rightJoin:
		lookup := plan.newLookup(plan.left.Len())
		rightRows, leftRows, err = pairedMatchingRows(plan.right, plan.left, lookup, true)
	case fullJoin:
		lookup := plan.newLookup(plan.right.Len())
		leftRows, rightRows, err = fullMatchingRows(plan.left, plan.right, lookup)
	case semiJoin, antiJoin:
		lookup := plan.newLookup(plan.right.Len())
		return left.Take(existenceRows(plan.left, plan.right, lookup, kind == semiJoin)), nil
	default:
		panic("dataframe: invalid join kind")
	}
	if err != nil {
		return Frame{}, err
	}
	return buildJoinFrame(left, right, plan, leftRows, rightRows, kind), nil
}

func resolveJoinPlan[K any](left, right Frame, key JoinKey[K]) (resolvedJoinPlan[K], error) {
	plan := resolvedJoinPlan[K]{
		left:      key.left,
		right:     key.right,
		newLookup: key.newLookup,
		projection: joinProjection{
			leftKeyIndex:  -1,
			rightKeyIndex: -1,
		},
	}
	if key.newLookup == nil {
		return resolvedJoinPlan[K]{}, fmt.Errorf("%w: join key has a nil hasher", ErrUnsupported)
	}
	if !key.usingStored {
		if key.left.Len() != left.Len() || key.right.Len() != right.Len() {
			return resolvedJoinPlan[K]{}, fmt.Errorf("%w: positional join keys have lengths %d and %d, want %d and %d", ErrRowCount, key.left.Len(), key.right.Len(), left.Len(), right.Len())
		}
		return plan, nil
	}
	if key.leftName == "" || key.rightName == "" || key.outputName == "" {
		return resolvedJoinPlan[K]{}, fmt.Errorf("%w: join key names must not be empty", ErrInvalidName)
	}
	leftKey, err := left.Column[K](key.leftName)
	if err != nil {
		return resolvedJoinPlan[K]{}, err
	}
	rightKey, err := right.Column[K](key.rightName)
	if err != nil {
		return resolvedJoinPlan[K]{}, err
	}
	plan.left = leftKey
	plan.right = rightKey
	plan.projection = joinProjection{
		leftKeyIndex:  left.columnIndex(key.leftName),
		rightKeyIndex: right.columnIndex(key.rightName),
		outputName:    key.outputName,
	}
	return plan, nil
}

func validateColumnNames(left, right Frame, projection joinProjection) error {
	names := make(map[string]struct{}, left.Width()+right.Width())
	for i, column := range left.columns {
		name := column.name
		if i == projection.leftKeyIndex {
			name = projection.outputName
		}
		if _, exists := names[name]; exists {
			return fmt.Errorf("%w: %q", ErrColumnConflict, name)
		}
		names[name] = struct{}{}
	}
	for i, column := range right.columns {
		if i == projection.rightKeyIndex {
			continue
		}
		name := column.name
		if _, exists := names[name]; exists {
			return fmt.Errorf("%w: %q", ErrColumnConflict, name)
		}
		names[name] = struct{}{}
	}
	return nil
}

func pairedMatchingRows[K any](probe, indexed series.Series[K], lookup joinLookup[K], keepUnmatched bool) ([]int, []int, error) {
	index := newJoinIndex(indexed, lookup)
	capacity := probe.Len()
	var precomputed []joinMatch
	if index.next != nil {
		var err error
		precomputed, capacity, err = precomputeJoinMatches(probe, index, keepUnmatched)
		if err != nil {
			return nil, nil, err
		}
	}
	probeRows := make([]int, 0, capacity)
	indexedRows := make([]int, 0, capacity)
	for probeRow := range probe.Len() {
		var matches joinMatch
		var found bool
		if precomputed != nil {
			matches = precomputed[probeRow]
			found = matches.count > 0
		} else {
			matches, found = index.matches(probe, probeRow)
		}
		if !found {
			if !keepUnmatched {
				continue
			}
			if len(probeRows) == math.MaxInt {
				return nil, nil, fmt.Errorf("%w: join result row count overflows int", ErrRowCount)
			}
			probeRows = append(probeRows, probeRow)
			indexedRows = append(indexedRows, -1)
			continue
		}
		if matches.count > math.MaxInt-len(probeRows) {
			return nil, nil, fmt.Errorf("%w: join result row count overflows int", ErrRowCount)
		}
		indexedRow := matches.first
		for remaining := matches.count; remaining > 0; remaining-- {
			probeRows = append(probeRows, probeRow)
			indexedRows = append(indexedRows, indexedRow)
			if remaining > 1 {
				indexedRow = index.next[indexedRow]
			}
		}
	}
	return probeRows, indexedRows, nil
}

func precomputeJoinMatches[K any](probe series.Series[K], index joinIndex[K], keepUnmatched bool) ([]joinMatch, int, error) {
	precomputed := make([]joinMatch, probe.Len())
	rowCount := 0
	for probeRow := range probe.Len() {
		matches, found := index.matches(probe, probeRow)
		if found {
			precomputed[probeRow] = matches
			if matches.count > math.MaxInt-rowCount {
				return nil, 0, fmt.Errorf("%w: join result row count overflows int", ErrRowCount)
			}
			rowCount += matches.count
			continue
		}
		if keepUnmatched {
			if rowCount == math.MaxInt {
				return nil, 0, fmt.Errorf("%w: join result row count overflows int", ErrRowCount)
			}
			rowCount++
		}
	}
	return precomputed, rowCount, nil
}

func fullMatchingRows[K any](left, right series.Series[K], lookup joinLookup[K]) ([]int, []int, error) {
	index := newJoinIndex(right, lookup)
	capacity := left.Len()
	capacity += min(right.Len(), math.MaxInt-capacity)
	leftRows := make([]int, 0, capacity)
	rightRows := make([]int, 0, capacity)
	matchedRight := make([]bool, right.Len())
	for leftRow := range left.Len() {
		matches, found := index.matches(left, leftRow)
		if !found {
			if len(leftRows) == math.MaxInt {
				return nil, nil, fmt.Errorf("%w: join result row count overflows int", ErrRowCount)
			}
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, -1)
			continue
		}
		if matches.count > math.MaxInt-len(leftRows) {
			return nil, nil, fmt.Errorf("%w: join result row count overflows int", ErrRowCount)
		}
		rightRow := matches.first
		for remaining := matches.count; remaining > 0; remaining-- {
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, rightRow)
			matchedRight[rightRow] = true
			if remaining > 1 {
				rightRow = index.next[rightRow]
			}
		}
	}
	for rightRow, matched := range matchedRight {
		if !matched {
			if len(leftRows) == math.MaxInt {
				return nil, nil, fmt.Errorf("%w: join result row count overflows int", ErrRowCount)
			}
			leftRows = append(leftRows, -1)
			rightRows = append(rightRows, rightRow)
		}
	}
	return leftRows, rightRows, nil
}

func existenceRows[K any](left, right series.Series[K], lookup joinLookup[K], keepMatches bool) []int {
	for row := range right.Len() {
		value, present := right.At(row)
		if present {
			lookup.Set(value, 0)
		}
	}
	rows := make([]int, 0, left.Len())
	for row := range left.Len() {
		value, present := left.At(row)
		if !present {
			if !keepMatches {
				rows = append(rows, row)
			}
			continue
		}
		_, found := lookup.Get(value)
		if found == keepMatches {
			rows = append(rows, row)
		}
	}
	return rows
}

func buildJoinFrame[K any](left, right Frame, plan resolvedJoinPlan[K], leftRows, rightRows []int, kind joinKind) Frame {
	var nullableLeftRows series.Series[int]
	var nullableRightRows series.Series[int]
	leftColumns := left.Width()
	rightColumns := right.Width()
	coalescedKey := plan.projection.leftKeyIndex >= 0
	if coalescedKey {
		leftColumns--
		rightColumns--
	}
	switch kind {
	case innerJoin:
	case leftJoin:
		if rightColumns > 0 {
			nullableRightRows = joinRowIndexes(rightRows)
		}
	case rightJoin:
		if leftColumns > 0 {
			nullableLeftRows = joinRowIndexes(leftRows)
		}
	case fullJoin:
		if leftColumns > 0 || coalescedKey {
			nullableLeftRows = joinRowIndexes(leftRows)
		}
		if rightColumns > 0 || coalescedKey {
			nullableRightRows = joinRowIndexes(rightRows)
		}
	case semiJoin, antiJoin:
		panic("dataframe: invalid projected join kind")
	default:
		panic("dataframe: invalid join kind")
	}

	columns := make([]column, 0, left.Width()+right.Width())
	for i, column := range left.columns {
		if i == plan.projection.leftKeyIndex {
			joined := joinedKey(plan, leftRows, rightRows, nullableLeftRows, nullableRightRows, kind)
			columns = append(columns, typedColumn(plan.projection.outputName, joined))
			continue
		}
		if kind == rightJoin || kind == fullJoin {
			column.nullable = true
			column.values = column.values.takeNullable(nullableLeftRows)
		} else {
			column.values = column.values.take(leftRows)
		}
		columns = append(columns, column)
	}
	for i, column := range right.columns {
		if i == plan.projection.rightKeyIndex {
			continue
		}
		if kind == leftJoin || kind == fullJoin {
			column.nullable = true
			column.values = column.values.takeNullable(nullableRightRows)
		} else {
			column.values = column.values.take(rightRows)
		}
		columns = append(columns, column)
	}
	return Frame{columns: columns, rowCount: len(leftRows)}
}

func joinedKey[K any](plan resolvedJoinPlan[K], leftRows, rightRows []int, nullableLeftRows, nullableRightRows series.Series[int], kind joinKind) series.Series[K] {
	switch kind {
	case innerJoin:
		left := plan.left.Take(leftRows)
		if plan.left.Nullable() || !plan.right.Nullable() {
			return left
		}
		return coalesceJoinKeys(left, plan.right.Take(rightRows))
	case leftJoin:
		return plan.left.Take(leftRows)
	case rightJoin:
		return plan.right.Take(rightRows)
	case fullJoin:
		result := coalesceJoinKeys(plan.left.TakeNullable(nullableLeftRows), plan.right.TakeNullable(nullableRightRows))
		if !plan.left.Nullable() && !plan.right.Nullable() {
			return result.DropNull()
		}
		return result
	default:
		panic("dataframe: invalid join kind")
	}
}

func coalesceJoinKeys[K any](left, right series.Series[K]) series.Series[K] {
	return left.Map2Cells(right, func(leftKey, rightKey series.Optional[K]) series.Optional[K] {
		if leftKey.Valid {
			return leftKey
		}
		return rightKey
	})
}

func joinRowIndexes(rows []int) series.Series[int] {
	return series.NewNullableFunc(len(rows), func(i int) (int, bool) {
		return rows[i], rows[i] >= 0
	})
}

func crossJoinLeftRows(leftLen, rightLen int) []int {
	rows := make([]int, leftLen*rightLen)
	offset := 0
	for leftRow := range leftLen {
		for i := range rightLen {
			rows[offset+i] = leftRow
		}
		offset += rightLen
	}
	return rows
}

func crossJoinRightRows(leftLen, rightLen int) []int {
	rows := make([]int, leftLen*rightLen)
	offset := 0
	for range leftLen {
		for rightRow := range rightLen {
			rows[offset] = rightRow
			offset++
		}
	}
	return rows
}
