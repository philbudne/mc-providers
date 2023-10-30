import logging
from typing import List
import os

from .exceptions import UnknownProviderException, UnavailableProviderException, APIKeyRequired
from .provider import ContentProvider
from .reddit import RedditPushshiftProvider
from .twitter import TwitterTwitterProvider
from .youtube import YouTubeYouTubeProvider
from .onlinenews import OnlineNewsWaybackMachineProvider, OnlineNewsMediaCloudProvider, OnlineNewsMediaCloudLegacyProvider

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
PLATFORM_SOURCE_MEDIA_CLOUD_LEGACY = 'mclegacy'
PLATFORM_SOURCE_MEDIA_CLOUD = "mediacloud"

NAME_SEPARATOR = "-"


def provider_name(platform: str, source: str) -> str:
    return platform + NAME_SEPARATOR + source


def available_provider_names() -> List[str]:
    platforms = []
    platforms.append(provider_name(PLATFORM_TWITTER, PLATFORM_SOURCE_TWITTER))
    platforms.append(provider_name(PLATFORM_YOUTUBE, PLATFORM_SOURCE_YOUTUBE))
    platforms.append(provider_name(PLATFORM_REDDIT, PLATFORM_SOURCE_PUSHSHIFT))
    platforms.append(provider_name(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_WAYBACK_MACHINE))
    platforms.append(provider_name(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_MEDIA_CLOUD))
    platforms.append(provider_name(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_MEDIA_CLOUD_LEGACY))
    return platforms


def provider_by_name(name: str, api_key: str = None) -> ContentProvider:
    parts = name.split(NAME_SEPARATOR)
    return provider_for(parts[0], parts[1], api_key)


def provider_for(platform: str, source: str, api_key: str = None) -> ContentProvider:
    """
    A factory method that returns the appropriate data provider. Throws an exception to let you know if the
    arguments are unsupported.
    :param platform: One of the PLATFORM_* constants above.
    :param source: One of the PLATFORM_SOURCE>* constants above.
    :return:
    """
    available = available_provider_names()
    if provider_name(platform, source) in available:
        if (platform == PLATFORM_TWITTER) and (source == PLATFORM_SOURCE_TWITTER):
            if api_key is None:
                raise APIKeyRequired(platform)
                
            platform_provider = TwitterTwitterProvider(api_key)
            
        elif (platform == PLATFORM_REDDIT) and (source == PLATFORM_SOURCE_PUSHSHIFT):
            platform_provider = RedditPushshiftProvider()
        
        elif (platform == PLATFORM_YOUTUBE) and (source == PLATFORM_SOURCE_YOUTUBE):
            if api_key is None:
                raise APIKeyRequired(platform)
                
            platform_provider = YouTubeYouTubeProvider(api_key)
        
        elif (platform == PLATFORM_ONLINE_NEWS) and (source == PLATFORM_SOURCE_WAYBACK_MACHINE):
            platform_provider = OnlineNewsWaybackMachineProvider()

        elif (platform == PLATFORM_ONLINE_NEWS) and (source == PLATFORM_SOURCE_MEDIA_CLOUD):
            platform_provider = OnlineNewsMediaCloudProvider()
        
        elif (platform == PLATFORM_ONLINE_NEWS) and (source == PLATFORM_SOURCE_MEDIA_CLOUD_LEGACY):
            if api_key is None:
                raise APIKeyRequired(platform)
            
            platform_provider = OnlineNewsMediaCloudLegacyProvider(api_key)
        
        else:
            raise UnknownProviderException(platform, source)
        
        return platform_provider
    else:
        raise UnavailableProviderException(platform, source)
