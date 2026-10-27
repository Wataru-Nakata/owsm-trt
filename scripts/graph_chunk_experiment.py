"""Profiler-only four-step graph experiment; production sessions/defaults are not modified."""

from owsm_trt.runtime import BeamSession, GreedySession, capture_graph


def run_chunks(self, c, pos, steps, done):
    if (self.chunk != self._chunk_width or getattr(self, 'diagnostics', False)
            or getattr(self, 'compact_tail', False)):
        return self._single_graph_run(self, c, pos, steps, done)
    while steps > 0:
        if pos >= c:
            c = self._grow(c, pos + 1)
        n = min(self.chunk, steps, c - pos)
        if n == self._chunk_width and c in self._chunk_graphs:
            self._chunk_graphs[c].replay()
        else:
            for _ in range(n):
                self.graphs[c].replay()
        steps -= n
        pos += n
        if self._all_done(done):
            break


class ChunkGreedySession(GreedySession):
    _run = run_chunks


class ChunkBeamSession(BeamSession):
    _run = run_chunks


def install_runner(session, width):
    # Instance-bound methods keep the session/graphs in a cycle. A later GC may destroy those
    # graphs during another capture, where CUDA forbids graph destruction. Keep only an unbound
    # fallback and use normal class binding so ordinary reference counting releases the graphs.
    original = type(session)
    session._single_graph_run = original._run
    session._chunk_width = width
    session.__class__ = {GreedySession: ChunkGreedySession, BeamSession: ChunkBeamSession}[original]


def prepare(session, width=4):
    """Cold-only, before request prefill: retain original graphs for boundaries and small buckets."""
    if width != 4:
        raise ValueError('this experiment supports four steps only')
    if hasattr(session, '_chunk_graph_info'):
        return session._chunk_graph_info
    reason = None
    if type(session) not in (GreedySession, BeamSession):
        reason = 'experiment supports the concrete greedy/beam session types only'
    elif session.device.type != 'cuda' or not session.use_cuda_graph:
        reason = 'CUDA graphs required'
    elif session.chunk != width:
        reason = 'preserve existing host-check interval'
    elif getattr(session, 'diagnostics', False) or getattr(session, 'compact_tail', False):
        reason = 'diagnostic/tail scheduling excluded from this experiment'
    info = dict(width=width, captures=[], fallback_reason=reason)
    session._chunk_graph_info = info
    if reason:
        return info
    if isinstance(session, GreedySession):
        session._prepare_plain()
    session._chunk_graphs = {}
    for c in session.buckets:
        # capture_graph executes two warmups and one capture. Do not overflow a tiny test bucket
        # or its sequence/history buffers during cold preparation.
        if min(c, session.capacity) <= 3 * width:
            continue
        session.pos.zero_()
        if hasattr(session, 'step'):
            session.step.zero_()

        def body(c=c):
            for _ in range(width):
                session._body(c)

        session._chunk_graphs[c] = capture_graph(body)
        info['captures'].append(c)
    if session._chunk_graphs:
        install_runner(session, width)
    else:
        info['fallback_reason'] = 'buckets too small for cold warmup/capture'
    return info
