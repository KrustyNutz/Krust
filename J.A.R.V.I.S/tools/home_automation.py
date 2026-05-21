import requests
from config.settings import HOME_ASSISTANT_URL, HOME_ASSISTANT_TOKEN


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
        "Content-Type": "application/json",
    }


def _base() -> str:
    return HOME_ASSISTANT_URL.rstrip("/") + "/api"


def ha_get_states(domain: str = "") -> list[dict]:
    if not HOME_ASSISTANT_URL:
        return [{"error": "HOME_ASSISTANT_URL not configured"}]
    resp = requests.get(f"{_base()}/states", headers=_headers(), timeout=10)
    resp.raise_for_status()
    states = resp.json()
    if domain:
        states = [s for s in states if s["entity_id"].startswith(domain + ".")]
    return [{"entity_id": s["entity_id"], "state": s["state"]} for s in states[:50]]


def ha_call_service(domain: str, service: str, entity_id: str, **kwargs) -> dict:
    if not HOME_ASSISTANT_URL:
        return {"error": "HOME_ASSISTANT_URL not configured"}
    data = {"entity_id": entity_id, **kwargs}
    resp = requests.post(
        f"{_base()}/services/{domain}/{service}",
        headers=_headers(),
        json=data,
        timeout=10,
    )
    resp.raise_for_status()
    return {"success": True, "entity_id": entity_id, "service": f"{domain}.{service}"}


def ha_get_entity(entity_id: str) -> dict:
    if not HOME_ASSISTANT_URL:
        return {"error": "HOME_ASSISTANT_URL not configured"}
    resp = requests.get(f"{_base()}/states/{entity_id}", headers=_headers(), timeout=10)
    if resp.status_code == 404:
        return {"error": f"Entity not found: {entity_id}"}
    resp.raise_for_status()
    return resp.json()


def register_ha_tools(registry):
    registry.register(
        "ha_get_states",
        "Get Home Assistant entity states. Filter by domain (light, switch, sensor, etc.)",
        {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Domain to filter by (light, switch, sensor, etc.)", "default": ""},
            },
        },
        ha_get_states,
    )

    registry.register(
        "ha_call_service",
        "Call a Home Assistant service. Examples: turn on a light, lock a door.",
        {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Service domain (light, switch, lock, etc.)"},
                "service": {"type": "string", "description": "Service name (turn_on, turn_off, toggle, etc.)"},
                "entity_id": {"type": "string", "description": "Target entity ID"},
            },
            "required": ["domain", "service", "entity_id"],
        },
        ha_call_service,
    )

    registry.register(
        "ha_get_entity",
        "Get full state and attributes of a specific Home Assistant entity.",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity ID (e.g. light.living_room)"},
            },
            "required": ["entity_id"],
        },
        ha_get_entity,
    )
