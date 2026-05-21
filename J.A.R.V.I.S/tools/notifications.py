def send_notification(title: str, message: str, urgency: str = "normal") -> str:
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="J.A.R.V.I.S",
            timeout=10 if urgency == "critical" else 5,
        )
        return f"Notification sent: '{title}'"
    except ImportError:
        # Fallback: try system notify-send on Linux
        import subprocess, shutil
        if shutil.which("notify-send"):
            urgency_map = {"low": "low", "normal": "normal", "critical": "critical"}
            subprocess.Popen([
                "notify-send",
                f"--urgency={urgency_map.get(urgency, 'normal')}",
                title,
                message,
            ])
            return f"Notification sent via notify-send: '{title}'"
        return f"Notification (no display): {title} — {message}"
    except Exception as e:
        return f"Notification failed: {e}"
