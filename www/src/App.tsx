import { Dispatch, ReactNode, useEffect, useMemo, useState } from "react"
import './App.css'
import { batched, interp, range, scan, shuffle } from "@rdub/base"
import { flatten, sum } from "lodash"
import { ClassName } from "@rdub/base/classname"

import seedrandom from 'seedrandom'

const getBarW = interp([ 40, .7 ], [ 100, .5 ])

function Bar({ i, n, x, w, h, }: { i: number, n: number, x: number, w: number, h: number }) {
  const fill = `hsl(${315 * i / n}, 100%, 50%)`
  return (
    <rect
      width={w}
      height={h}
      x={x}
      fill={fill}
    />
  )
}

function getBarWX({ nBars, nChunks, }: {
  nBars: number
  nChunks: number
}): {
  w: number
  x: (i: number, chunkIdx: number) => number
} {
  const w = getBarW(nBars)
  const nEmpties = nChunks - 1
  const N = nBars + nEmpties
  const nGaps = N - 1
  const totalBarW = w * N
  const gapW = (100 - totalBarW) / nGaps
  const x = (i: number, chunkIdx: number) => {
    const idx = i + chunkIdx
    return idx * (w + gapW)
  }
  return { w, x }
}

export function Bars({ groups, n, y = 0, h, className }: { groups: number[][], n?: number, y?: number, h: number } & ClassName) {
  const groupLens = groups.map(g => g.length)
  n = n ?? sum(groupLens)
  const startIdxs = scan(groupLens, (acc, x) => acc + x, 0)

  const bars = getBarWX({ nBars: n, nChunks: groups.length })
  return (
    <g className={className} transform={`translate(0, ${y ?? 0})`}>{
      groups.map((group, chunkIdx) => {
        return <g className={"shuffleChunk"} key={chunkIdx}>{
          group.map((i, idx) => {
            const xIdx = startIdxs[chunkIdx] + idx
            return <Bar key={i} i={i} n={n} x={bars.x(xIdx, chunkIdx)} w={bars.w} h={h} />
          })
        }</g>
      })
    }</g>
  )
}

function Number({ label, min, state: [ val, set ] }: { label: ReactNode, min?: number, state: [ number, Dispatch<number> ] }) {
  const [ str, setStr ] = useState(val.toString())
  const [ err, setErr ] = useState(false)
  useEffect(() => {
    const val = parseInt(str)
    const err = isNaN(val) || (min !== undefined && val < min)
    setErr(err)
    if (!err) {
      set(val)
    }
  }, [ str, setErr, ])
  return <label className={"number"}>
    <span>{label}</span>
    <input
      type={"number"}
      className={err ? "err" : undefined}
      value={str}
      width={5}
      min={min}
      onChange={e => {
        setStr(e.target.value)
      }}
    />
  </label>
}

function App() {
  const [ n, setN ] = useState(100)
  const [ seed, setSeed ] = useState(0)
  const [ shuffleChunkSize, setShuffleChunkSize ] = useState(5)
  const [ ioBatchSize, setIoBatchSize ] = useState(20)
  const [ gpuBatchSize, setGpuBatchSize ] = useState(4)

  const rng = useMemo((() => seedrandom(`${seed}`)), [ seed ])

  const barH = 3
  const barsGap = 3
  const idxs = useMemo(() => range(n), [n])
  const shuffleChunks = useMemo(() => shuffle(batched(idxs, shuffleChunkSize), rng), [ idxs, shuffleChunkSize, rng ])
  const ioBatches = useMemo(() => batched(flatten(shuffleChunks), ioBatchSize).map(ioBatch => shuffle(ioBatch, rng)), [shuffleChunks, ioBatchSize, rng])
  const gpuBatches = useMemo(() => batched(flatten(ioBatches), gpuBatchSize), [ioBatches, gpuBatchSize])
  const groups = [
    [idxs],
    shuffleChunks,
    ioBatches,
    gpuBatches
  ]
  const rowH = barH + barsGap
  const H = groups.length * rowH - barsGap
  return (
    <>
      <div className="container">
        <svg viewBox={`0 0 100 ${H}`}>{
          groups.map((groups, idx) => <Bars key={idx} groups={groups} y={idx * rowH} h={barH} />)
        }
        </svg>
        <div className={"controls"}>
          <Number label={"N"} min={1} state={[ n, setN ]} />
          <Number label={"Seed"} state={[ seed, setSeed ]} />
          <Number label={"Shuffle chunk"} min={1} state={[ shuffleChunkSize, setShuffleChunkSize ]} />
          <Number label={"IO batch"} min={1} state={[ ioBatchSize, setIoBatchSize ]} />
          <Number label={"GPU batch"} min={1} state={[ gpuBatchSize, setGpuBatchSize ]} />
        </div>
      </div>
    </>
  )
}

export default App
