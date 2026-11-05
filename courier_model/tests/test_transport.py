import unittest

from courier_model import main as cm


class TransportOptionMappingTest(unittest.TestCase):
    def test_auto_only(self):
        self.assertEqual(cm.transport_allowed_vehicles(cm.TRANSPORT_AUTO), {cm.AUTO})

    def test_bike_only_includes_moto_variants(self):
        for code in (
            cm.TRANSPORT_BIKE,
            cm.TRANSPORT_MOTO,
            cm.TRANSPORT_BIKE_MOTO,
        ):
            self.assertEqual(cm.transport_allowed_vehicles(code), {cm.BIKE})

    def test_auto_bike_includes_moto_mix(self):
        for code in (
            cm.TRANSPORT_AUTO_BIKE,
            cm.TRANSPORT_AUTO_MOTO,
            cm.TRANSPORT_AUTO_BIKE_MOTO,
        ):
            self.assertEqual(
                cm.transport_allowed_vehicles(code),
                {cm.AUTO, cm.BIKE},
            )


if __name__ == "__main__":
    unittest.main()
