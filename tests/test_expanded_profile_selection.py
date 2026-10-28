"""Small manifest fixtures only; no model/audio operations."""

import pytest
from profile_expanded_greedy import select_inputs


def rows():
    return [dict(key=f'{lang}-{bucket}-{i}',lang=lang,dataset='speech',long_form=False,
                 screening_overlap=False,seconds=seconds,duration_bucket=bucket)
            for lang in ('eng','jpn') for bucket,seconds in [('short',2),('long',20)] for i in range(4)]


def test_selection_is_order_independent_and_balances_duration_strata():
    source=rows()
    result=select_inputs(source,4,30)
    assert result==select_inputs(list(reversed(source)),4,30)
    for group in result.values():
        assert len(group)==4 and len({r['key'] for r in group})==4
        assert sum(r['seconds']==2 for r in group)==2


def test_selection_excludes_screening_longform_nonspeech_and_overlong_inputs():
    source=rows()
    for index,change in enumerate([dict(screening_overlap=True),dict(long_form=True),
                                    dict(dataset='stress'),dict(seconds=31)]):
        source.append(source[0]|change|dict(key=f'excluded-{index}'))
    assert select_inputs(source,4,30)==select_inputs(rows(),4,30)


def test_selection_rejects_insufficient_or_duplicate_inputs():
    with pytest.raises(ValueError,match='insufficient'):
        select_inputs(rows(),9,30)
    with pytest.raises(ValueError,match='duplicate'):
        select_inputs(rows()+rows()[:1],4,30)


def test_subset_preserves_exact_per_language_selection_and_canonical_order():
    full=select_inputs(rows(),4,30)
    assert select_inputs(rows(),4,30,['jpn'])=={'jpn':full['jpn']}
    assert select_inputs(rows(),4,30,['jpn','eng'])==full


@pytest.mark.parametrize('languages',[[],['eng','eng'],['absent']])
def test_invalid_subset_cannot_silently_drop_requested_languages(languages):
    with pytest.raises(ValueError,match='language subset'):
        select_inputs(rows(),4,30,languages)
