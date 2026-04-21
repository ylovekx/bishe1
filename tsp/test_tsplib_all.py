from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
TSPLIB_DIR = REPO_ROOT / 'data' / 'tsplib'
SOLUTIONS_FILE = TSPLIB_DIR / 'solutions'

# Ensure we import the 2-opt capable ACO from tsp_nls rather than tsp/aco.py.
if 'aco' in sys.modules:
    del sys.modules['aco']
sys.path.insert(0, str(REPO_ROOT / 'tsp_nls'))
from aco import ACO as ACO2OPT  # type: ignore  # noqa: E402
sys.path.pop(0)

sys.path.insert(0, str(CURRENT_DIR))
from net_orig import Net as NetOrig  # type: ignore  # noqa: E402
from net_gat import Net as NetGAT  # type: ignore  # noqa: E402
sys.path.pop(0)

EPS = 1e-10


@dataclass
class TsplibProblem:
    name: str
    dimension: int
    edge_weight_type: str
    edge_weight_format: Optional[str]
    coords: Optional[np.ndarray]
    distances: np.ndarray


def parse_solutions(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not path.exists():
        return mapping
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        name, value = line.split(':', 1)
        value = value.strip().split()[0]
        try:
            mapping[name.strip()] = int(value)
        except ValueError:
            continue
    return mapping


def _parse_header(lines: Sequence[str]) -> Tuple[Dict[str, str], int]:
    header: Dict[str, str] = {}
    idx = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in {'NODE_COORD_SECTION', 'EDGE_WEIGHT_SECTION', 'DISPLAY_DATA_SECTION'}:
            return header, idx
        if ':' in stripped:
            key, value = stripped.split(':', 1)
            header[key.strip().upper()] = value.strip()
        else:
            parts = stripped.split(None, 1)
            if len(parts) == 2:
                header[parts[0].strip().upper()] = parts[1].strip()
    return header, idx + 1


def _parse_node_coords(lines: Sequence[str], start_idx: int, n: int) -> np.ndarray:
    coords: List[Tuple[float, float]] = []
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped or stripped == 'EOF':
            break
        parts = stripped.split()
        if len(parts) >= 3:
            coords.append((float(parts[1]), float(parts[2])))
        if len(coords) >= n:
            break
    if len(coords) != n:
        raise ValueError(f'Expected {n} coordinates, got {len(coords)}')
    return np.asarray(coords, dtype=np.float64)


def _expand_explicit_matrix(values: Sequence[float], n: int, fmt: str) -> np.ndarray:
    fmt = fmt.upper()
    mat = np.zeros((n, n), dtype=np.float64)
    k = 0

    if fmt == 'FULL_MATRIX':
        arr = np.asarray(values, dtype=np.float64)
        if arr.size != n * n:
            raise ValueError(f'FULL_MATRIX expects {n*n} values, got {arr.size}')
        return arr.reshape(n, n)

    if fmt == 'UPPER_ROW':
        for i in range(n):
            for j in range(i + 1, n):
                mat[i, j] = values[k]
                mat[j, i] = values[k]
                k += 1
        return mat

    if fmt == 'UPPER_DIAG_ROW':
        for i in range(n):
            for j in range(i, n):
                mat[i, j] = values[k]
                mat[j, i] = values[k]
                k += 1
        return mat

    if fmt == 'LOWER_DIAG_ROW':
        for i in range(n):
            for j in range(i + 1):
                mat[i, j] = values[k]
                mat[j, i] = values[k]
                k += 1
        return mat

    raise NotImplementedError(f'Unsupported EDGE_WEIGHT_FORMAT: {fmt}')


def _parse_explicit_matrix(lines: Sequence[str], start_idx: int, n: int, fmt: str) -> np.ndarray:
    values: List[float] = []
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped or stripped == 'EOF':
            break
        values.extend(float(x) for x in stripped.split())
    return _expand_explicit_matrix(values, n, fmt)


def _tsplib_nint(x: float) -> int:
    return int(x + 0.5)


def _geo_to_rad(x: float) -> float:
    deg = int(x)
    minutes = x - deg
    return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0


def build_distance_matrix(coords: np.ndarray, edge_weight_type: str) -> np.ndarray:
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    typ = edge_weight_type.upper()
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            if typ == 'EUC_2D':
                dij = _tsplib_nint(math.sqrt(dx * dx + dy * dy))
            elif typ == 'CEIL_2D':
                dij = math.ceil(math.sqrt(dx * dx + dy * dy))
            elif typ == 'ATT':
                rij = math.sqrt((dx * dx + dy * dy) / 10.0)
                tij = _tsplib_nint(rij)
                dij = tij + 1 if tij < rij else tij
            elif typ == 'GEO':
                lati = _geo_to_rad(coords[i, 0])
                longi = _geo_to_rad(coords[i, 1])
                latj = _geo_to_rad(coords[j, 0])
                longj = _geo_to_rad(coords[j, 1])
                q1 = math.cos(longi - longj)
                q2 = math.cos(lati - latj)
                q3 = math.cos(lati + latj)
                dij = int(6378.388 * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1.0)
            else:
                raise NotImplementedError(f'Unsupported EDGE_WEIGHT_TYPE: {typ}')
            dist[i, j] = dist[j, i] = float(dij)
    return dist


def classical_mds(distances: np.ndarray, n_components: int = 2) -> np.ndarray:
    n = distances.shape[0]
    J = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    B = -0.5 * J @ (distances ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    pos = np.maximum(eigvals[:n_components], 0.0)
    vecs = eigvecs[:, :n_components]
    coords = vecs * np.sqrt(pos)
    if coords.shape[1] < n_components:
        coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])))
    return coords.astype(np.float64)


def normalize_features(coords: np.ndarray) -> np.ndarray:
    mn = coords.min(axis=0, keepdims=True)
    mx = coords.max(axis=0, keepdims=True)
    scale = np.where((mx - mn) < 1e-6, 1.0, (mx - mn))
    return (coords - mn) / scale


def load_tsplib_problem(path: Path) -> TsplibProblem:
    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    header, section_idx = _parse_header(lines)
    name = header.get('NAME', path.stem)
    dimension = int(header['DIMENSION'])
    edge_weight_type = header.get('EDGE_WEIGHT_TYPE', 'EUC_2D').upper()
    edge_weight_format = header.get('EDGE_WEIGHT_FORMAT')

    coords: Optional[np.ndarray] = None
    distances: np.ndarray

    section = lines[section_idx].strip()
    if edge_weight_type == 'EXPLICIT':
        fmt = edge_weight_format or 'FULL_MATRIX'
        distances = _parse_explicit_matrix(lines, section_idx, dimension, fmt)
    else:
        if section != 'NODE_COORD_SECTION':
            try:
                section_idx = next(i for i, line in enumerate(lines) if line.strip() == 'NODE_COORD_SECTION')
            except StopIteration as exc:
                raise ValueError(f'Cannot find NODE_COORD_SECTION in {path}') from exc
        coords = _parse_node_coords(lines, section_idx, dimension)
        distances = build_distance_matrix(coords, edge_weight_type)

    np.fill_diagonal(distances, 1e9)
    return TsplibProblem(
        name=name,
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        edge_weight_format=edge_weight_format,
        coords=coords,
        distances=distances,
    )


def build_pyg(features: np.ndarray, distances: np.ndarray, k_sparse: int, device: torch.device) -> Tuple[Data, torch.Tensor]:
    n = distances.shape[0]
    k = max(1, min(k_sparse, n - 1))
    dist_tensor = torch.tensor(distances, dtype=torch.float32, device=device)
    topk_values, topk_indices = torch.topk(dist_tensor, k=k, dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n, device=device), repeats=k),
        topk_indices.reshape(-1),
    ])
    edge_attr = topk_values.reshape(-1, 1)
    x = torch.tensor(features, dtype=torch.float32, device=device)
    pyg = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg, dist_tensor


def model_bucket(n: int) -> int:
    if n < 50:
        return 20
    if n < 200:
        return 100
    return 500


def default_k(n: int) -> int:
    bucket = model_bucket(n)
    return {20: 10, 100: 20, 500: 50}[bucket]


def resolve_checkpoint(kind: str, bucket: int) -> Path:
    kind = kind.lower()
    if kind == 'orig':
        candidates = [
            CURRENT_DIR / f'tsp{bucket}_orig.pt',
            REPO_ROOT / 'pretrained' / 'tsp' / f'tsp{bucket}.pt',
        ]
    elif kind == 'gat':
        candidates = [CURRENT_DIR / f'tsp{bucket}_gatv2.pt']
    else:
        raise ValueError(kind)

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f'No checkpoint found for {kind} bucket={bucket}: {candidates}')


def load_model(kind: str, bucket: int, device: torch.device):
    ckpt = resolve_checkpoint(kind, bucket)
    model = NetOrig().to(device) if kind == 'orig' else NetGAT().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def _maybe_sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_vanilla_aco(distances: torch.Tensor, n_ants: int, iterations: int, k_sparse: int, device: torch.device) -> Tuple[float, float]:
    _maybe_sync(device)
    start = time.time()
    aco = ACO2OPT(n_ants=n_ants, distances=distances, device=device)
    if hasattr(aco, 'sparsify'):
        aco.sparsify(min(k_sparse, len(distances) - 1))
    cost = float(aco.run(iterations))
    _maybe_sync(device)
    return cost, time.time() - start


def run_deepaco(model, pyg_data: Data, distances: torch.Tensor, n_ants: int, iterations: int, device: torch.device, heuristic_power: float = 1.0) -> Tuple[float, float, float]:
    _maybe_sync(device)
    start = time.time()
    with torch.no_grad():
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
        if abs(heuristic_power - 1.0) > 1e-12:
            heu_mat = heu_mat ** heuristic_power
    _maybe_sync(device)
    infer_time = time.time() - start

    _maybe_sync(device)
    solve_start = time.time()
    aco = ACO2OPT(n_ants=n_ants, heuristic=heu_mat, distances=distances, device=device)
    cost = float(aco.run(iterations))
    _maybe_sync(device)
    total = infer_time + (time.time() - solve_start)
    return cost, infer_time, total


def select_features(problem: TsplibProblem) -> np.ndarray:
    if problem.coords is not None:
        return normalize_features(problem.coords)
    # Fallback for EXPLICIT instances: embed the exact matrix into 2D pseudo-coordinates.
    return normalize_features(classical_mds(problem.distances, n_components=2))


def evaluate_instance(path: Path, bks_map: Dict[str, int], orig_cache: Dict[int, Tuple[torch.nn.Module, Path]], gat_cache: Dict[int, Tuple[torch.nn.Module, Path]], args, device: torch.device) -> Dict[str, object]:
    problem = load_tsplib_problem(path)
    bucket = model_bucket(problem.dimension)
    k_sparse = args.k_sparse if args.k_sparse is not None else default_k(problem.dimension)

    features = select_features(problem)
    pyg_data, distances = build_pyg(features, problem.distances, k_sparse, device)

    if bucket not in orig_cache:
        orig_cache[bucket] = load_model('orig', bucket, device)
    if bucket not in gat_cache:
        gat_cache[bucket] = load_model('gat', bucket, device)
    orig_model, orig_ckpt = orig_cache[bucket]
    gat_model, gat_ckpt = gat_cache[bucket]

    bks = bks_map.get(problem.name)

    aco_cost, aco_time = run_vanilla_aco(distances, args.n_ants, args.iterations, k_sparse, device)
    orig_cost, orig_infer, orig_total = run_deepaco(orig_model, pyg_data, distances, args.n_ants, args.iterations, device, heuristic_power=args.orig_power)
    gat_cost, gat_infer, gat_total = run_deepaco(gat_model, pyg_data, distances, args.n_ants, args.iterations, device, heuristic_power=args.gat_power)

    def gap(cost: float) -> Optional[float]:
        if bks is None:
            return None
        return (cost - bks) / bks * 100.0

    return {
        'instance': problem.name,
        'n': problem.dimension,
        'bucket': bucket,
        'edge_weight_type': problem.edge_weight_type,
        'edge_weight_format': problem.edge_weight_format or '',
        'k_sparse': k_sparse,
        'bks': bks if bks is not None else '',
        'aco_cost': round(aco_cost, 4),
        'aco_time_s': round(aco_time, 4),
        'aco_gap_pct': '' if gap(aco_cost) is None else round(gap(aco_cost), 4),
        'orig_cost': round(orig_cost, 4),
        'orig_infer_s': round(orig_infer, 4),
        'orig_total_s': round(orig_total, 4),
        'orig_gap_pct': '' if gap(orig_cost) is None else round(gap(orig_cost), 4),
        'orig_ckpt': str(orig_ckpt.relative_to(REPO_ROOT)),
        'gat_cost': round(gat_cost, 4),
        'gat_infer_s': round(gat_infer, 4),
        'gat_total_s': round(gat_total, 4),
        'gat_gap_pct': '' if gap(gat_cost) is None else round(gap(gat_cost), 4),
        'gat_ckpt': str(gat_ckpt.relative_to(REPO_ROOT)),
    }


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        print('No instances were evaluated.')
        return
    print('\n=== Summary ===')
    print(f'Total instances: {len(rows)}')
    for key in ('aco_cost', 'orig_cost', 'gat_cost', 'aco_time_s', 'orig_total_s', 'gat_total_s'):
        vals = [float(r[key]) for r in rows]
        print(f'{key}: mean={sum(vals)/len(vals):.4f}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate TSPLIB instances with vanilla ACO, original DeepACO and 8-layer GATv2 DeepACO.')
    parser.add_argument('--tsplib-dir', type=Path, default=TSPLIB_DIR)
    parser.add_argument('--solutions', type=Path, default=SOLUTIONS_FILE)
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'results' / 'tsplib_all_results.csv')
    parser.add_argument('--iterations', type=int, default=100, help='ACO iterations per instance')
    parser.add_argument('--n-ants', type=int, default=100, help='Number of ants during evaluation')
    parser.add_argument('--k-sparse', type=int, default=None, help='Override KNN sparsification for all instances')
    parser.add_argument('--max-n', type=int, default=None, help='Only evaluate instances with n <= max_n')
    parser.add_argument('--limit', type=int, default=None, help='Only run the first N instances after sorting by name')
    parser.add_argument('--gat-power', type=float, default=0.5, help='Power sharpening applied to the GAT heuristic matrix')
    parser.add_argument('--orig-power', type=float, default=1.0, help='Optional power applied to the original DeepACO heuristic matrix')
    parser.add_argument('--cpu', action='store_true', help='Force CPU even when CUDA is available')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    bks_map = parse_solutions(args.solutions)

    files = sorted(args.tsplib_dir.glob('*.tsp'), key=lambda p: p.name.lower())
    if args.max_n is not None:
        filtered = []
        for path in files:
            try:
                problem = load_tsplib_problem(path)
            except Exception as exc:
                print(f'[skip] {path.name}: failed to inspect dimension ({exc})')
                continue
            if problem.dimension <= args.max_n:
                filtered.append(path)
        files = filtered
    if args.limit is not None:
        files = files[: args.limit]

    print(f'Device: {device}')
    print(f'TSPLIB directory: {args.tsplib_dir}')
    print(f'Instances to evaluate: {len(files)}')
    print(f'Iterations: {args.iterations}, ants: {args.n_ants}, k_sparse: {args.k_sparse or "bucket-default"}')
    if args.max_n is None:
        print('Warning: evaluating very large TSPLIB instances may take a long time or run out of memory.')

    rows: List[Dict[str, object]] = []
    orig_cache: Dict[int, Tuple[torch.nn.Module, Path]] = {}
    gat_cache: Dict[int, Tuple[torch.nn.Module, Path]] = {}

    for idx, path in enumerate(files, start=1):
        print(f'[{idx}/{len(files)}] {path.name}')
        try:
            row = evaluate_instance(path, bks_map, orig_cache, gat_cache, args, device)
            rows.append(row)
            print(
                f"  n={row['n']} | ACO {row['aco_cost']} ({row['aco_time_s']}s) | "
                f"Orig {row['orig_cost']} ({row['orig_total_s']}s) | "
                f"GAT {row['gat_cost']} ({row['gat_total_s']}s)"
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f'  failed: {exc}')

    write_csv(rows, args.output)
    print(f'\nSaved CSV to: {args.output}')
    print_summary(rows)


if __name__ == '__main__':
    main()
