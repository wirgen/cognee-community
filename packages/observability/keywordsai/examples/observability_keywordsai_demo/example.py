import asyncio
import json
import os
from typing import Any

import cognee_community_observability_keywordsai  # noqa: F401  (patches Cognee)
from cognee import prune, visualize_graph
from cognee.low_level import DataPoint, setup
from cognee.modules.data.methods import load_or_create_datasets
from cognee.modules.observability.get_observe import get_observe
from cognee.modules.users.methods import get_default_user
from cognee.pipelines import Task, run_tasks
from cognee.tasks.storage import add_data_points as _add_data_points

observe = get_observe()


class Person(DataPoint):
    name: str
    metadata: dict[str, Any] = {"index_fields": ["name"]}


class Department(DataPoint):
    name: str
    employees: list[Person]
    metadata: dict[str, Any] = {"index_fields": ["name"]}


class CompanyType(DataPoint):
    name: str = "Company"
    metadata: dict[str, Any] = {"index_fields": ["name"]}


class Company(DataPoint):
    name: str
    departments: list[Department]
    is_type: CompanyType
    metadata: dict[str, Any] = {"index_fields": ["name"]}


# ---------- Task: build DataPoints ---------- #
@observe  # task span
def ingest_files(data: list[Any]):
    people_dp, dept_dp, company_dp = {}, {}, {}
    for item in data:
        people, companies = item["people"], item["companies"]
        for p in people:
            person = people_dp[p["name"]] = Person(name=p["name"])
            dept_dp.setdefault(
                p["department"], Department(name=p["department"], employees=[])
            ).employees.append(person)

        ctype = CompanyType()
        for c in companies:
            comp = company_dp[c["name"]] = Company(name=c["name"], departments=[], is_type=ctype)
            for d in c["departments"]:
                comp.departments.append(dept_dp.setdefault(d, Department(name=d, employees=[])))
    return company_dp.values()


# Wrap to trace
add_data_points = observe(_add_data_points)


# ---------- Workflow ---------- #
@observe(workflow=True)  # workflow span
async def main():
    await prune.prune_data()
    await prune.prune_system(metadata=True)
    await setup()

    user = await get_default_user()
    ds = await load_or_create_datasets(["test_dataset"], [], user)

    base = os.path.dirname(__file__)
    companies = json.load(open(os.path.join(base, "companies.json")))
    people = json.load(open(os.path.join(base, "people.json")))
    data = [{"companies": companies, "people": people}]

    pipeline = run_tasks(
        [Task(ingest_files), Task(add_data_points)],
        dataset_id=ds[0].id,
        data=data,
    )
    async for s in pipeline:
        print(s)

    html_out = os.path.join(base, ".artifacts/graph_visualization.html")
    await visualize_graph(html_out)
    return html_out


if __name__ == "__main__":
    asyncio.run(main())
