"""Exercise build cleanup with tiny fake installers/compressors; never build or load an environment."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


@pytest.mark.parametrize('kind,failure',[
    ('main',None),('ref',None),('main','install'),('main','archive'),('main','metadata'),
])
def test_private_workdir_removed_and_results_preserved(tmp_path,kind,failure):
    repo = tmp_path/'repo'
    (repo/'scripts').mkdir(parents=True)
    script = repo/'scripts/build_env_tar.sh'
    shutil.copyfile(Path(__file__).parents[1]/'scripts/build_env_tar.sh',script)
    (repo/'tools/espnet_ref').mkdir(parents=True)
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    cp = bin_dir/'cp'
    cp.write_text('''#!/bin/bash
if [ "${MOCK_FAILURE:-}" = metadata ] && [[ "${@: -1}" = *.requirements.txt ]]; then
    exit 9
fi
exec /bin/cp "$@"
''')
    cp.chmod(0o755)
    uv = bin_dir/'uv'
    uv.write_text('''#!/bin/bash
set -eu
if [ "$1 ${2:-}" = "python install" ]; then
    mkdir -p "$UV_PYTHON_INSTALL_DIR/python/bin" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"
    printf '#!/bin/bash\\necho fake-python-check\\n' > "$UV_PYTHON_INSTALL_DIR/python/bin/python3"
    chmod +x "$UV_PYTHON_INSTALL_DIR/python/bin/python3"
elif [ "$1 ${2:-}" = "python find" ]; then
    echo "$UV_PYTHON_INSTALL_DIR/python/bin/python3"
elif [ "$1" = export ]; then
    while [ "$1" != -o ]; do shift; done
    echo fake-dependency > "$2"
elif [ "$1" = pip ] && [ "${MOCK_FAILURE:-}" = install ]; then
    exit 7
fi
''')
    zstd = bin_dir/'zstd'
    zstd.write_text('''#!/bin/bash
set -eu
while [ "$1" != -o ]; do shift; done
cat > "$2"
if [ "${MOCK_FAILURE:-}" = archive ]; then exit 8; fi
''')
    uv.chmod(0o755)
    zstd.chmod(0o755)
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    owned = scratch/'unrelated'
    owned.write_text('preserve me')
    out = repo/'.cache/envs'
    out.mkdir(parents=True)
    name = 'owsm-env' if kind=='main' else 'espnet-ref'
    archive = out/f'{name}.tar.zst'
    archive.write_bytes(b'previous archive')
    env = os.environ | dict(PBS_JOBID='synthetic-test',PBS_LOCALDIR=str(scratch),BUILD_ROOT=str(scratch),
                           PATH=str(bin_dir)+':'+os.environ['PATH'],MOCK_FAILURE=failure or '')
    result = subprocess.run(['bash',str(script),kind],env=env,capture_output=True,text=True)
    assert (result.returncode==0)==(failure is None),result.stderr
    assert list(scratch.iterdir())==[owned]
    if failure=='metadata':
        assert not (out/f'{name}.requirements.txt').exists()
    else:
        assert (out/f'{name}.requirements.txt').read_text()=='fake-dependency\n'
    assert not list(out.glob('*.tmp.*'))
    if failure in ('install','archive'):
        assert archive.read_bytes()==b'previous archive'
    else:
        with tarfile.open(archive) as data:
            assert data.getmember(name+'/bin/python3').isfile()
