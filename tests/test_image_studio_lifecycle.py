from __future__ import annotations

from io import BytesIO
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from openagent.image_studio import ImageStudio
from openagent.routes.image_studio_routes import ROUTES
from openagent.storage import OpenAgentStore


def png(width: int = 8, height: int = 4) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), (250, 250, 250)).save(output, format="PNG")
    return output.getvalue()


class ImageStudioLifecycleTests(unittest.TestCase):
    def test_real_transform_versions_metadata_album_and_undo(self) -> None:
        with TemporaryDirectory() as temp:
            studio = ImageStudio(OpenAgentStore(temp))
            asset = studio.upload_asset(filename="test.png", content=png(), project="core")
            rotated = studio.rotate(asset["id"], degrees=90)
            self.assertEqual((rotated["asset"]["width"], rotated["asset"]["height"]), (4, 8))
            updated = studio.edit_metadata(asset["id"], tags=["demo"], favorite=True)
            self.assertTrue(updated["metadata"]["favorite"])
            second = studio.upload_asset(filename="second.png", content=png(5, 5), project="core")
            batch = studio.batch_tag([asset["id"], second["id"]], ["reviewed", "Demo"])
            self.assertEqual(batch["count"], 2)
            self.assertIn("reviewed", studio.get_asset(second["id"])["tags"])
            album = studio.create_album("Tests", project="core")
            studio.add_to_album(album["id"], asset["id"])
            self.assertEqual(studio.albums(project="core")[0]["asset_count"], 1)
            restored = studio.undo(asset["id"])
            self.assertEqual((restored["asset"]["width"], restored["asset"]["height"]), (8, 4))

    def test_routes_cover_reversible_studio(self) -> None:
        paths = {path for _, path in ROUTES}
        self.assertIn("/api/gallery/{id}/transform", paths)
        self.assertIn("/api/gallery/{id}/undo", paths)
        self.assertIn("/api/gallery/albums", paths)
        self.assertIn("/api/gallery/batch-tags", paths)


if __name__ == "__main__":
    unittest.main()
