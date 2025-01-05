import logging
from typing import List, Optional

from .exceptions import UnknownProviderException, UnavailableProviderException, APIKeyRequired
from .provider import ContentProvider, DEFAULT_TIMEOUT, set_default_timeout
from .reddit import RedditPushshiftProvider
from .twitter import TwitterTwitterProvider
from .youtube import YouTubeYouTubeProvider
from .onlinenews import OnlineNewsWaybackMachineProvider, OnlineNewsMediaCloudESProvider

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

NAME_SEPARATOR = "-"

def provider_name(platform: str, source: str) -> str:
    return platform + NAME_SEPARATOR + source


def available_provider_names() -> List[str]:
    return [
        provider_name(PLATFORM_TWITTER, PLATFORM_SOURCE_TWITTER),
        provider_name(PLATFORM_YOUTUBE, PLATFORM_SOURCE_YOUTUBE),
        provider_name(PLATFORM_REDDIT, PLATFORM_SOURCE_PUSHSHIFT),
        provider_name(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_WAYBACK_MACHINE),
        provider_name(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_MEDIA_CLOUD)
    ]

# api_key, base_url, timeout, caching, etc
# must now always be passed by keyword
def provider_by_name(name: str, **kwargs) -> ContentProvider:
    """
    For kwargs, see provider_for
    """
    platform, source = name.split(NAME_SEPARATOR)
    return provider_for(platform, source, **kwargs)


def provider_for(platform: str, source: str, **kwargs) -> ContentProvider:
    """
    A factory method that returns the appropriate data provider. Throws an exception to let you know if the
    platform/source arguments are unsupported.
    :param platform: One of the PLATFORM_* constants above.
    :param source: One of the PLATFORM_SOURCE>* constants above.

    All providers support kwargs:
    :param caching: zero to disable in-library caching
    :param timeout: override the default timeout for the provider (in seconds)

    Providers may support (among others):
    :param api_key: The API key needed to access the provider.
    :param base_url: For custom integrations you can provide an alternate base URL for the provider's API server
    :param session_id: String that identifies client session
    :param software_id: String that identifies client software

    :return: the appropriate ContentProvider subclass
    """
    available = available_provider_names()
    platform_provider: ContentProvider
    if provider_name(platform, source) in available:
        api_key = kwargs.pop("api_key", None)
        if (platform == PLATFORM_TWITTER) and (source == PLATFORM_SOURCE_TWITTER):
            if api_key is None:
                raise APIKeyRequired(platform)

            platform_provider = TwitterTwitterProvider(api_key, **kwargs)

        elif (platform == PLATFORM_REDDIT) and (source == PLATFORM_SOURCE_PUSHSHIFT):
            platform_provider = RedditPushshiftProvider(**kwargs)

        elif (platform == PLATFORM_YOUTUBE) and (source == PLATFORM_SOURCE_YOUTUBE):
            if api_key is None:
                raise APIKeyRequired(platform)

            platform_provider = YouTubeYouTubeProvider(api_key, **kwargs)
        
        elif (platform == PLATFORM_ONLINE_NEWS) and (source == PLATFORM_SOURCE_WAYBACK_MACHINE):
            platform_provider = OnlineNewsWaybackMachineProvider(**kwargs)

        elif (platform == PLATFORM_ONLINE_NEWS) and (source == PLATFORM_SOURCE_MEDIA_CLOUD):
            platform_provider = OnlineNewsMediaCloudESProvider(**kwargs)

        else:
            raise UnknownProviderException(platform, source)

        return platform_provider
    else:
        raise UnavailableProviderException(platform, source)
