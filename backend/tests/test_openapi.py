"""OpenAPI contract と project license metadata を検証する。"""

from sphere_reconstruct.main import app


def test_openapi_declares_license_and_purpose_scoped_mask_routes():
    schema = app.openapi()

    assert schema["info"]["license"] == {
        "name": "GPL-3.0-or-later",
        "identifier": "GPL-3.0-or-later",
    }
    assert "/api/projects/{project_id}/masks/{purpose}" in schema["paths"]
    prepared = schema["paths"]["/api/projects/{project_id}/prepared-mask"]["get"]
    assert any(parameter["name"] == "purpose" for parameter in prepared["parameters"])
