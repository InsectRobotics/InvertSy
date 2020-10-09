from datetime import datetime
from pytz import timezone


class Observer(object):
    def __init__(self, lon=None, lat=None, date=datetime.now()):
        self._lon = lon
        self._lat = lat
        self._date = date
        self._city = None
        h_loc = date.hour
        h_gmt = date.astimezone(timezone("GMT")).hour
        self._tz = h_loc - h_gmt
        self.on_change = None

    @property
    def lon(self):
        """
        The longitude of the observer.
        """
        return self._lon

    @lon.setter
    def lon(self, value):
        self._lon = float(value)
        if self.on_change is not None:
            self.on_change()

    @property
    def lat(self):
        """
        The latitude of the observer.
        """
        return self._lat

    @lat.setter
    def lat(self, value):
        self._lat = float(value)
        if self.on_change is not None:
            self.on_change()

    @property
    def timezone(self):
        return self._tz

    @property
    def date(self):
        """
        The date and time in the current position.
        """
        return self._date

    @date.setter
    def date(self, value):
        self._date = value
        if self.on_change is not None:
            self.on_change()

    @property
    def city(self):
        return self._city

    def __repr__(self):
        return "Observer(lon='%.6f', lat='%.6f', %sdate='%s')" % (
            self.lon, self.lat, ("city='%s', " % self.city) if self._city is not None else "", str(self._date))


def get_seville_observer():
    sev = Observer()
    sev.lat = '37.392509'
    sev.lon = '-5.983877'
    sev._city = "Seville"

    return sev


def get_edinburgh_observer():
    edi = Observer()
    edi.lat = '55.946388'
    edi.lon = '-3.200000'
    edi._city = "Edinburgh"

    return edi


def get_live_observer():
    import requests
    import json

    send_url = "http://api.ipstack.com/check?access_key=9d6917440142feeccd73751e2f2124dc"
    geo_req = requests.get(send_url)
    geo_json = json.loads(geo_req.text)

    obs = Observer(lon=geo_json['longitude'], lat=geo_json['latitude'], date=datetime.now())
    obs._city = geo_json['city']
    return obs


if __name__ == '__main__':
    print(get_live_observer())
