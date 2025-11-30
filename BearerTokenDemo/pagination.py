import requests

def list_all_issues(owner: str, repo: str, state: str = "open", token: str = None):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"

    # Add an auth header *only if* user provides a GitHub token
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    params = {"state": state, "per_page": 50, "page": 1}
    all_issues: list[dict] = []

    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=5)

        # Stop if GitHub says: "You asked beyond the last page" or "Rate limit exceeded"
        if resp.status_code in (403, 422):
            print("Stopped: rate limit reached or no more pages.")
            break

        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):  # GitHub sometimes returns error JSON
            break

        if not data:
            break  # no more items

        all_issues.extend(data)
        params["page"] += 1

    return all_issues


# Example usage (without token)
issues = list_all_issues("octocat", "Hello-World", "open")

for issue in issues:
    print(issue.get("title", "No title"))
