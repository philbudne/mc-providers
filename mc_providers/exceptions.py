
class ProviderException(Exception):
    pass


class UnsupportedOperationException(ProviderException):
    pass


class UnknownProviderException(ProviderException):
    def __init__(self, platform: str, source: str):
        super().__init__("Unknown provider {} from {}".format(platform, source))


class MissingRequiredValue(ProviderException):
    def __init__(self, name: str, keyword: str):
        super().__init__(f"provider {name} requires {keyword}")


class QueryingEverythingUnsupportedQuery(ProviderException):
    def __init__(self):
        super().__init__("Can't query everything")


# backwards compatibility:
APIKeyRequired = MissingRequiredValue
UnavailableProviderException = UnknownProviderException
