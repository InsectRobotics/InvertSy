__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from datetime import datetime
from pytz import timezone
from copy import copy

import numpy as np


class Observer(object):
    def __init__(self, lon=None, lat=None, date=datetime.now(), city=None, inrad=True):
        if lon is not None and lat is not None:
            self._lon = float(lon) if inrad else np.deg2rad(float(lon))
            self._lat = float(lat) if inrad else np.deg2rad(float(lat))
        else:
            self._lon = lon
            self._lat = lat
        self._date = date
        self._city = city
        h_loc = date.hour
        h_gmt = date.astimezone(timezone("GMT")).hour
        self._tzinfo = date.tzinfo
        self._tz = h_loc - h_gmt
        self.on_change = None
        self.__inrad = inrad

    @property
    def lon(self) -> float:
        """
        The longitude of the observer.
        """
        return self._lon if self.__inrad else np.rad2deg(self._lon)

    @lon.setter
    def lon(self, value: {float, int, str}):
        self._lon = float(value) if self.__inrad else np.deg2rad(float(value))
        if self.on_change is not None:
            self.on_change()

    @property
    def lat(self) -> float:
        """
        The latitude of the observer.
        """
        return self._lat if self.__inrad else np.rad2deg(self._lat)

    @lat.setter
    def lat(self, value: {float, int, str}):
        self._lat = float(value) if self.__inrad else np.deg2rad(float(value))
        if self.on_change is not None:
            self.on_change()

    @property
    def tzgmt(self) -> int:
        return self._tz

    @property
    def timezone(self):
        return self._tzinfo

    @property
    def date(self) -> datetime:
        """
        The date and time in the current position.
        """
        return self._date

    @date.setter
    def date(self, value: datetime):
        self._date = value
        if self.on_change is not None:
            self.on_change()

    @property
    def city(self) -> str:
        return self._city

    def copy(self):
        return copy(self)

    def __copy__(self):
        return Observer(lon=copy(self.lon), lat=copy(self.lat), inrad=copy(self.__inrad),
                        date=copy(self.date), city=copy(self.city))

    def __repr__(self):
        return "Observer(lon='%.6f', lat='%.6f', %sdate='%s', timezone='%s')" % (
            self.lon, self.lat, ("city='%s', " % self.city) if self._city is not None else "",
            str(self._date), self.timezone)


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


# if __name__ == '__main__':
#     print(get_live_observer())


# 1.79m sit, 3.48m pet, 2.48m 3yl, 2.18m ore, 2.98m xry
# 1.96m sit, 1.49m pet, 1.86m 3yl, 2.23m ore, 1.12m xry
# 1.56m sit, 3.04m pet, 2.17m 3yl, 1.91m ore, 1.30m xry
# 5.31m sit, 8.01m pet, 6.51m 3yl, 6.24m ore, 5.40m xry
