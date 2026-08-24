from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True, order=True)
class SensitivityChannel:
    """
    ENDF-aware identifier for one sensitivity-profile axis entry.

    ``average_MF``/``average_MT`` identify the mean (best-estimate) nuclear-data quantity 
    to which the sensitivity refers. 
    ``covariance_MF``/``covariance_MT`` identify
    the primary covariance channel used by sandwich calculations. 
    Additional valid covariance MTs, such as MF34/MT2 for the first elastic Legendre
    moment, can be stored in ``covariance_MT_aliases``. ``L`` is used for
    angular-distribution Legendre moments.
    """

    average_MF: Optional[int]
    average_MT: Optional[int]
    covariance_MF: Optional[int] = None
    covariance_MT: Optional[int] = None
    L: Optional[int] = None
    name: Optional[str] = field(default=None, compare=False)
    covariance_MT_aliases: Tuple[int, ...] = field(default=(),
                                                  compare=False)

    @property
    def MT(self):
        """Return the average ENDF MT number."""
        return self.average_MT

    @classmethod
    def from_alias(cls, alias):
        """Build a channel from a known human/readers' label."""
        label = " ".join(str(alias).split()).lower()
        if label in _ALIASES:
            return _ALIASES[label]
        if label.startswith("xs "):
            mt = int(label.split()[1])
            return cls.from_endf(average_MF=3, average_MT=mt, name=label)
        raise ValueError(f"Unknown sensitivity channel alias {alias!r}.")

    @classmethod
    def from_endf(cls, *, average_MF=None, average_MT=None, covariance_MF=None, 
                  covariance_MT=None, L=None, name=None):
        """
        Build a channel from average-side or covariance-side ENDF identifiers.

        If the supplied identifiers match a registered ENDF quantity, the
        registered channel is returned. Otherwise a generic channel is created
        when enough information is available.
        """
        has_average_pair = average_MF is not None and average_MT is not None
        has_covariance_pair = (
            covariance_MF is not None and covariance_MT is not None)
        if has_average_pair or has_covariance_pair:
            matches = []
            for channel in set(_ALIASES.values()):
                if average_MF is not None and channel.average_MF != average_MF:
                    continue
                if average_MT is not None and channel.average_MT != average_MT:
                    continue
                if covariance_MF is not None and channel.covariance_MF != covariance_MF:
                    continue
                if covariance_MT is not None and covariance_MT not in channel.covariance_MTs:
                    continue
                if L is not None and channel.L != L:
                    continue
                matches.append(channel)

            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    "ENDF identifiers match multiple sensitivity channels: "
                    f"{sorted(matches)}. Provide L or a more specific channel."
                )

        if average_MF is None and covariance_MF == 33:
            average_MF = 3
        if average_MT is None and covariance_MT is not None:
            average_MT = covariance_MT
        if covariance_MF is None and average_MF is not None:
            covariance_MF = _COVARIANCE_MF_BY_AVERAGE_MF.get(average_MF)
        if covariance_MT is None and average_MT is not None:
            covariance_MT = average_MT
        if name is None and average_MT is not None:
            name = f"MT {average_MT}"

        return cls( average_MF=average_MF, average_MT=average_MT, 
                    covariance_MF=covariance_MF, covariance_MT=covariance_MT, 
                    L=L, name=name, )

    def matches_covariance(self, MF, MT):
        """Return whether this channel can be paired with covariance MF/MT."""
        return self.covariance_MF == int(MF) and int(MT) in self.covariance_MTs

    @property
    def covariance_MTs(self):
        """Return the primary covariance MT plus accepted aliases."""
        if self.covariance_MT is None:
            return ()
        return (self.covariance_MT, ) + tuple(self.covariance_MT_aliases)


_COVARIANCE_MF_BY_AVERAGE_MF = {
                                    1: 31,
                                    3: 33,
                                    4: 34,
                                    5: 35,
                                }

NUBAR_TOTAL = SensitivityChannel(1, 452, 31, 452, name="nubar total")
NUBAR_PROMPT = SensitivityChannel(1, 456, 31, 456, name="nubar prompt")
NUBAR_DELAYED = SensitivityChannel(1, 455, 31, 455, name="nubar delayed")
CHI_TOTAL = SensitivityChannel(None, None, None, None, name="chi total")
CHI_PROMPT = SensitivityChannel(5, 18, 35, 18, name="chi prompt")
CHI_DELAYED = SensitivityChannel(5, 455, 35, 455, name="chi delayed")
ELASTIC_LEGENDRE_P1 = SensitivityChannel(4, 2, 34, 251, L=1, name="elastic Legendre moment 1", 
                                         covariance_MT_aliases=(2, ))
ELASTIC_LEGENDRE_P2 = SensitivityChannel(4, 2, 34, 2, L=2, name="elastic Legendre moment 2")

_ALIASES = {
            "nubar total": NUBAR_TOTAL,
            "nubar prompt": NUBAR_PROMPT,
            "nubar delayed": NUBAR_DELAYED,
            "chi total": CHI_TOTAL,
            "chi prompt": CHI_PROMPT,
            "chi delayed": CHI_DELAYED,
            "ela leg mom 1": ELASTIC_LEGENDRE_P1,
            "ela leg mom 2": ELASTIC_LEGENDRE_P2,
            }
