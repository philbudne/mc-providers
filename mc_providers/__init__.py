import logging
from typing import Any, List, NamedTuple, Optional

from .exceptions import UnknownProviderException, MissingRequiredValue
from .provider import ContentProvider, DEFAULT_TIMEOUT, set_default_timeout
from .reddit import RedditPushshiftProvider
from .twitter import TwitterTwitterProvider
from .youtube import YouTubeYouTubeProvider
from .onlinenews import OnlineNewsWaybackMachineProvider, OnlineNewsMediaCloudProvider, OnlineNewsMediaCloudESProvider

logger = logging.getLogger(__name__)

# static list matching topics/info results
PLATFORM_TWITTER = 'twitter'
PLATFORM_REDDIT = 'reddit'
PLATFORM_YOUTUBE = 'youtube'
PLATFORM_ONLINE_NEWS = 'onlinenews'

# static list matching topics/info results
PLATFORM_SOURCE_PUSHSHIFT = 'pushshift'
PLATFORM_SOURCE_TWITTER = 'twitter'
PLATFORM_SOURCE_YOUTUBE = 'youtube'
PLATFORM_SOURCE_WAYBACK_MACHINE = 'waybackmachine'
PLATFORM_SOURCE_MEDIA_CLOUD = "mediacloud"
PLATFORM_SOURCE_MEDIA_CLOUD_OLD = "mediacloud-old"


NAME_SEPARATOR = "-"

class _PT(NamedTuple):          # provider tuple
    cls: type[ContentProvider]
    required: list[str] = []


def provider_name(platform: str, source: str) -> str:
    return platform + NAME_SEPARATOR + source

_KEY_REQ = ["api_key"]
_PROVIDER_MAP: dict[str, _PT] = {
    provider_name(PLATFORM_TWITTER, PLATFORM_SOURCE_TWITTER): _PT(TwitterTwitterProvider, _KEY_REQ),
    provider_name(PLATFORM_YOUTUBE, PLATFORM_SOURCE_YOUTUBE): _PT(RedditPushshiftProvider),
    provider_name(PLATFORM_REDDIT, PLATFORM_SOURCE_PUSHSHIFT): _PT(YouTubeYouTubeProvider, _KEY_REQ),
    provider_name(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_WAYBACK_MACHINE): _PT(OnlineNewsWaybackMachineProvider),
    provider_name(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_MEDIA_CLOUD_OLD): _PT(OnlineNewsMediaCloudProvider),
    provider_name(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_MEDIA_CLOUD): _PT(OnlineNewsMediaCloudESProvider),
}

_PROVIDER_NAMES: List[str] = list(_PROVIDER_MAP.keys())

def available_provider_names() -> List[str]:
    # called from frontend/index.html view, so pre-calculated
    return _PROVIDER_NAMES

def provider_for(platform: str, source: str, **kwargs: Any) -> ContentProvider:
    """
    :param platform: One of the PLATFORM_* constants above.
    :param source: One of the PLATFORM_SOURCE_* constants above.

    see provider_by_name for kwargs
    """
    return provider_by_name(provider_name(platform, source), **kwargs)


def provider_by_name(name: str, **kwargs: Any) -> ContentProvider:
    """
    A factory method that returns the appropriate data provider. Throws an exception to let you know if the
    platform/source arguments are unsupported.

    All providers support kwargs:
    :param caching: zero to disable in-library caching
    :param timeout: override the default timeout for the provider (in seconds)

    Providers may support (among others):
    :param api_key: The API key needed to access the provider (may be required)
    :param base_url: For custom integrations you can provide an alternate base URL for the provider's API server
    :param session_id: String that identifies client session
    :param software_id: String that identifies client software

    :return: the appropriate ContentProvider subclass
    """
    platform_provider: ContentProvider

    if name not in _PROVIDER_MAP:
        platform, source = name.split(NAME_SEPARATOR, 1)
        raise UnknownProviderException(platform, source)

    pt = _PROVIDER_MAP[name]
    for required in pt.required:
        if not kwargs.get(required):
            raise MissingRequiredValue(platform, required)

    return pt.cls(**kwargs)
