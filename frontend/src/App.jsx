import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import TerrainScene from "./components/TerrainScene.jsx";
import StoryPanel from "./components/StoryPanel.jsx";
import DriveHud from "./components/DriveHud.jsx";
import { getTerrainMeta, planRoute, getRlStatus } from "./api/client.js";
import { pathTitle, pathShort } from "./utils/pathLabels.js";
import { planSegmentData, segmentSlopeDegPlan, rlTrailTotalMeters } from "./utils/driveMetrics.js";
import "./App.css";

function summarizePlan(plan, meta) {
  if (!plan || !meta || !plan.waypoints?.length) return null;
  const { segLens, total } = planSegmentData(plan, meta);
  if (!segLens.length || total <= 0) return null;

  let maxDropM = 0;
  let maxSlopeDeg = 0;
  let hazardAccum = 0;

  for (let i = 0; i < plan.waypoints.length - 1; i++) {
    const e0 = plan.elevations_m[i];
    const e1 = plan.elevations_m[i + 1];
    const drop = e0 - e1;
    if (drop > maxDropM) maxDropM = drop;
    const slopeDeg = segmentSlopeDegPlan(plan, meta, i);
    if (slopeDeg > maxSlopeDeg) maxSlopeDeg = slopeDeg;
    const roughness =
      Math.abs((plan.elevations_m[i + 2] ?? e1) - 2 * e1 + e0) +
      Math.abs(e1 - 2 * e0 + (plan.elevations_m[i - 1] ?? e0));
    const hazard = Math.min(1, slopeDeg / 35 + roughness / 8);
    hazardAccum += hazard * (segLens[i] || 0);
  }

  const avgHazard = hazardAccum / total || 0;
  const riskScore = Math.max(0, Math.min(100, avgHazard * 100));
  const energyPerKm =
    plan.energy_score != null && plan.total_distance_m > 1
      ? plan.energy_score / (plan.total_distance_m / 1000)
      : null;

  return {
    totalDistanceM: plan.total_distance_m,
    energyScore: plan.energy_score,
    energyPerKm,
    maxDropM,
    maxSlopeDeg,
    riskScore,
  };
}

function summarizeRl(trail, meta) {
  if (!trail || trail.length < 2 || !meta) return null;
  const totalDistanceM = rlTrailTotalMeters(trail, meta);
  if (!totalDistanceM) return null;

  let maxDropM = 0;
  let maxSlopeDeg = 0;
  let hazardAccum = 0;

  for (let i = 0; i < trail.length - 1; i++) {
    const a = trail[i];
    const b = trail[i + 1];
    const e0 = a.elevM ?? 0;
    const e1 = b.elevM ?? 0;
    const drop = e0 - e1;
    if (drop > maxDropM) maxDropM = drop;

    const mpp = meta.meters_per_pixel;
    const dx = (b.x - a.x) * mpp;
    const dz = (b.z - a.z) * mpp;
    const horiz = Math.hypot(dx, dz);
    const slopeDeg =
      horiz > 1e-6 ? (Math.atan2(Math.abs(e1 - e0), horiz) * 180) / Math.PI : 0;
    if (slopeDeg > maxSlopeDeg) maxSlopeDeg = slopeDeg;

    const ePrev = i > 0 ? trail[i - 1].elevM ?? e0 : e0;
    const eNext = i < trail.length - 2 ? trail[i + 2].elevM ?? e1 : e1;
    const roughness =
      Math.abs(eNext - 2 * e1 + e0) + Math.abs(e1 - 2 * e0 + ePrev);
    const hazard = Math.min(1, slopeDeg / 35 + roughness / 8);
    const segLen = Math.hypot(dx, dz, e1 - e0) || 0;
    hazardAccum += hazard * segLen;
  }

  const avgHazard = hazardAccum / totalDistanceM || 0;
  const riskScore = Math.max(0, Math.min(100, avgHazard * 100));

  return {
    totalDistanceM,
    maxDropM,
    maxSlopeDeg,
    riskScore,
  };
}

export default function App() {
  const [mode, setMode] = useState("story");
  const [meta, setMeta] = useState(null);
  const [err, setErr] = useState(null);
  const [rlStatus, setRlStatus] = useState(null);
  const [start, setStart] = useState(null);
  const [goal, setGoal] = useState(null);
  const [paths, setPaths] = useState({ astar: null, straight: null });
  const [loading, setLoading] = useState(false);
  const [rlTrail, setRlTrail] = useState([]);
  const [rlBusy, setRlBusy] = useState(false);
  const [showCostmap, setShowCostmap] = useState(false);
  const [cameraMode, setCameraMode] = useState("orbit");
  const [flyPath, setFlyPath] = useState(null);
  const [driveMetrics, setDriveMetrics] = useState(null);
  const rlWsRef = useRef(null);

  useEffect(() => {
    getTerrainMeta()
      .then(setMeta)
      .catch((e) => setErr(String(e)));
    getRlStatus()
      .then(setRlStatus)
      .catch(() => {});
  }, []);

  useEffect(() => {
    return () => {
      if (rlWsRef.current) {
        rlWsRef.current.close();
        rlWsRef.current = null;
      }
    };
  }, []);

  const onTerrainPick = useCallback(
    (col, row) => {
      if (!meta || mode !== "explore") return;
      if (!start) {
        setStart({ col, row });
        setGoal(null);
        setPaths({ astar: null, straight: null });
        setRlTrail([]);
        return;
      }
      if (!goal) {
        setGoal({ col, row });
        return;
      }
      setStart({ col, row });
      setGoal(null);
      setPaths({ astar: null, straight: null });
      setRlTrail([]);
    },
    [meta, start, goal, mode]
  );

  const runPlans = async () => {
    if (!start || !goal) return;
    setLoading(true);
    setErr(null);
    try {
      const [astar, straight] = await Promise.all([
        planRoute({
          start_col: start.col,
          start_row: start.row,
          goal_col: goal.col,
          goal_row: goal.row,
          planner: "astar",
        }),
        planRoute({
          start_col: start.col,
          start_row: start.row,
          goal_col: goal.col,
          goal_row: goal.row,
          planner: "straight",
        }),
      ]);
      setPaths({ astar, straight });
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  const streamRl = useCallback(async () => {
    if (!start || !goal || !meta) return;
    if (rlWsRef.current) {
      rlWsRef.current.close();
      rlWsRef.current = null;
    }
    setFlyPath(null);
    setCameraMode("orbit");
    setDriveMetrics(null);
    setRlBusy(true);
    setRlTrail([]);
    setErr(null);
    const yScale = Math.min(160, Math.max(14, meta.elevation_range_m * 0.55));
    const er = meta.elevation_range_m + 1e-9;
    const CURVATURE = 0.00011;
    const toWorld = (col, row, elevM) => {
      const x = col - meta.width / 2;
      const z = row - meta.height / 2;
      const normH = (elevM - meta.elevation_min) / er;
      return {
        x,
        z,
        y: normH * yScale - (x * x + z * z) * CURVATURE + 1.5,
        elevM,
        col,
        row,
      };
    };

    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${wsProto}//${window.location.host}/ws/rl-episode`);
    rlWsRef.current = ws;

    const trail = [];
    ws.onmessage = (ev) => {
      if (rlWsRef.current !== ws) return;
      const msg = JSON.parse(ev.data);
      if (msg.pos && msg.elevation_m != null) {
        const [col, row] = msg.pos;
        trail.push(toWorld(col, row, msg.elevation_m));
        setRlTrail([...trail]);
      }
      if (msg.done === true) {
        setRlBusy(false);
        if (rlWsRef.current === ws) {
          ws.close(1000, "episode_complete");
        }
      }
    };
    ws.onerror = () => setErr("WebSocket error (is the API running on :8000?)");
    ws.onclose = () => {
      if (rlWsRef.current === ws) rlWsRef.current = null;
      setRlBusy(false);
    };
    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          start: [start.col, start.row],
          goal: [goal.col, goal.row],
        })
      );
    };
  }, [start, goal, meta]);

  const startDrive = useCallback(
    (pathName) => {
      setCameraMode("follow");
      setFlyPath(pathName);
    },
    []
  );

  const onFlyComplete = useCallback(() => {
    setFlyPath(null);
    setCameraMode("orbit");
    setDriveMetrics(null);
  }, []);

  const onDriveMetrics = useCallback((m) => {
    setDriveMetrics(m);
  }, []);

  const insight = useMemo(() => {
    if (!paths?.astar || !paths?.straight) return null;
    const a = paths.astar;
    const s = paths.straight;
    const longer = a.total_distance_m > s.total_distance_m * 1.01;
    const lowerEnergy = a.energy_score < s.energy_score * 0.98;
    return { longer, lowerEnergy, a, s };
  }, [paths]);

  const comparison = useMemo(() => {
    if (!meta) return { astar: null, straight: null, rl: null };
    return {
      astar: paths.astar ? summarizePlan(paths.astar, meta) : null,
      straight: paths.straight ? summarizePlan(paths.straight, meta) : null,
      rl: rlTrail.length >= 2 ? summarizeRl(rlTrail, meta) : null,
    };
  }, [paths, rlTrail, meta]);

  const hint = useMemo(() => {
    if (mode === "story") return "";
    if (!start) return "Click the terrain to place a start point";
    if (!goal) return "Click again to place the goal";
    return "Run planners or drive the rover";
  }, [mode, start, goal]);

  const hasPaths = paths.astar || paths.straight;

  const activePlanForDrive = useMemo(() => {
    if (!flyPath || flyPath === "rl") return null;
    if (flyPath === "astar") return paths.astar;
    if (flyPath === "straight") return paths.straight;
    return null;
  }, [flyPath, paths]);

  return (
    <div className="app">
      <aside className="panel">
        <h1>Mars Route Zero</h1>
        {mode === "story" ? (
          <StoryPanel onEnterExplore={() => setMode("explore")} />
        ) : (
          <>
            <button
              type="button"
              className="ghost"
              onClick={() => {
                if (rlWsRef.current) {
                  rlWsRef.current.close();
                  rlWsRef.current = null;
                }
                setRlBusy(false);
                setMode("story");
                setCameraMode("orbit");
                setFlyPath(null);
                setDriveMetrics(null);
              }}
            >
              &larr; Story
            </button>

            <p className="explore-thesis">
              Same HiRISE grid for every run: compare how far each path goes and
              how much energy it costs — that tradeoff is what rover ops wrestle
              with daily.
            </p>

            <p className="hint step-hint">{hint}</p>

            {start && (
              <div className="marker-row">
                <span className="marker-dot start-dot" />
                <span className="marker-label">
                  Start ({start.col}, {start.row})
                </span>
              </div>
            )}
            {goal && (
              <div className="marker-row">
                <span className="marker-dot goal-dot" />
                <span className="marker-label">
                  Goal ({goal.col}, {goal.row})
                </span>
              </div>
            )}

            <div className="row">
              <button
                type="button"
                disabled={!start || !goal || loading}
                onClick={runPlans}
              >
                {loading
                  ? "Planning\u2026"
                  : "Plan optimized vs shortest path"}
              </button>
              <button
                type="button"
                disabled={!start || !goal || rlBusy}
                onClick={streamRl}
              >
                {rlBusy ? "Streaming\u2026" : "Stream RL"}
              </button>
            </div>

            <label className="toggle">
              <input
                type="checkbox"
                checked={showCostmap}
                onChange={(e) => setShowCostmap(e.target.checked)}
              />
              Cost overlay
            </label>

            {(hasPaths || rlTrail.length >= 2) && (
              <div className="stats">
                <div className="stats-header">
                  <h3>Path comparison</h3>
                  <button
                    type="button"
                    className="cam-toggle"
                    onClick={() =>
                      setCameraMode((m) => (m === "orbit" ? "follow" : "orbit"))
                    }
                  >
                    {cameraMode === "orbit" ? "Follow cam" : "Orbit cam"}
                  </button>
                </div>

                {paths.astar && (
                  <div className="stat-line">
                    <span className="legend-dot" style={{ background: "#ffffff" }} />
                    <span className="stat-text">
                      <strong>{pathTitle.astar}</strong>{" "}
                      {paths.astar.total_distance_m.toFixed(0)}m &middot; energy{" "}
                      {paths.astar.energy_score.toFixed(0)} kJ
                      {comparison.astar && (
                        <span className="stat-sub">
                          max drop {comparison.astar.maxDropM.toFixed(0)}m &middot; max slope{" "}
                          {comparison.astar.maxSlopeDeg.toFixed(1)}° &middot; est fuel{" "}
                          {comparison.astar.energyPerKm != null
                            ? `${comparison.astar.energyPerKm.toFixed(0)} kJ/km`
                            : "—"}{" "}
                          &middot; risk {comparison.astar.riskScore.toFixed(0)}/100
                        </span>
                      )}
                    </span>
                    <button
                      type="button"
                      className="drive-btn"
                      disabled={flyPath != null}
                      onClick={() => startDrive("astar")}
                    >
                      Drive
                    </button>
                  </div>
                )}

                {paths.straight && (
                  <div className="stat-line">
                    <span className="legend-dot" style={{ background: "#00d4ff" }} />
                    <span className="stat-text">
                      <strong>{pathTitle.straight}</strong>{" "}
                      {paths.straight.total_distance_m.toFixed(0)}m &middot; energy{" "}
                      {paths.straight.energy_score.toFixed(0)} kJ
                      {comparison.straight && (
                        <span className="stat-sub">
                          max drop {comparison.straight.maxDropM.toFixed(0)}m &middot; max
                          slope {comparison.straight.maxSlopeDeg.toFixed(1)}° &middot; est
                          fuel{" "}
                          {comparison.straight.energyPerKm != null
                            ? `${comparison.straight.energyPerKm.toFixed(0)} kJ/km`
                            : "—"}{" "}
                          &middot; risk {comparison.straight.riskScore.toFixed(0)}/100
                        </span>
                      )}
                    </span>
                    <button
                      type="button"
                      className="drive-btn"
                      disabled={flyPath != null}
                      onClick={() => startDrive("straight")}
                    >
                      Drive
                    </button>
                  </div>
                )}

                {rlTrail.length >= 2 && (
                  <div className="stat-line">
                    <span className="legend-dot" style={{ background: "#ff6b35" }} />
                    <span className="stat-text">
                      <strong>{pathTitle.rl}</strong> rollout ({rlTrail.length} steps)
                      {comparison.rl && (
                        <span className="stat-sub">
                          span {comparison.rl.totalDistanceM.toFixed(0)}m &middot; max drop{" "}
                          {comparison.rl.maxDropM.toFixed(0)}m &middot; max slope{" "}
                          {comparison.rl.maxSlopeDeg.toFixed(1)}° &middot; risk{" "}
                          {comparison.rl.riskScore.toFixed(0)}/100
                        </span>
                      )}
                    </span>
                    <button
                      type="button"
                      className="drive-btn"
                      disabled={flyPath != null}
                      onClick={() => startDrive("rl")}
                    >
                      Drive
                    </button>
                  </div>
                )}

                {insight && hasPaths && (
                  <p className="insight">
                    {insight.longer && insight.lowerEnergy
                      ? "The optimized path trades extra distance for lower energy cost \u2014 the tradeoff NASA optimizes for."
                      : insight.lowerEnergy
                        ? "The physics- and energy-optimized path beats the shortest segment line on energy."
                        : "Steeper terrain raises energy even when the shortest path looks temptingly direct."}
                  </p>
                )}
              </div>
            )}

            <button
              type="button"
              className="ghost"
              onClick={() => {
                if (rlWsRef.current) {
                  rlWsRef.current.close();
                  rlWsRef.current = null;
                }
                setRlBusy(false);
                setStart(null);
                setGoal(null);
                setPaths({ astar: null, straight: null });
                setRlTrail([]);
                setFlyPath(null);
                setCameraMode("orbit");
                setDriveMetrics(null);
              }}
            >
              Clear picks
            </button>

            {rlStatus && (
              <p className="small muted">
                Policy:{" "}
                {rlStatus.ppo_zip_exists ? "PPO found" : "heuristic"} &middot;
                ONNX {rlStatus.onnx_exists ? "ok" : "not exported"}
              </p>
            )}
            {err && <p className="err">{err}</p>}
          </>
        )}
      </aside>
      <main className="scene-wrap">
        {meta && (
          <TerrainScene
            meta={meta}
            paths={mode === "explore" ? paths : { astar: null, straight: null }}
            rlTrail={mode === "explore" ? rlTrail : []}
            showCostmap={mode === "explore" && showCostmap}
            onPick={onTerrainPick}
            pickEnabled={mode === "explore"}
            startPos={mode === "explore" ? start : null}
            goalPos={mode === "explore" ? goal : null}
            cameraMode={cameraMode}
            flyPath={flyPath}
            onFlyComplete={onFlyComplete}
            onDriveMetrics={onDriveMetrics}
            activePlan={activePlanForDrive}
            rlTrailSamples={mode === "explore" ? rlTrail : []}
            autoRotate={mode === "story"}
          />
        )}
        {!meta && !err && <div className="loading">Loading terrain\u2026</div>}
        <DriveHud metrics={driveMetrics} />
        {flyPath && (
          <div className="fly-overlay">
            <span className="fly-label">
              Driving{" "}
              {flyPath === "astar"
                ? pathShort.astar
                : flyPath === "straight"
                  ? pathShort.straight
                  : pathShort.rl}{" "}
              path\u2026
            </span>
            <button
              type="button"
              className="fly-stop"
              onClick={() => {
                setFlyPath(null);
                setCameraMode("orbit");
                setDriveMetrics(null);
              }}
            >
              Stop
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
