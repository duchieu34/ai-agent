from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from statistics import median
from pathlib import Path
from typing import Any

from .base import DomainAdapter
from ..models import DatasetContext


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class ChallengeAdapter(DomainAdapter):
    mode = "challenge"
    id_label = "TransactionID"
    entity_label = "transaction"

    def __init__(self, max_candidate_pool: int) -> None:
        super().__init__(max_candidate_pool=max_candidate_pool)
        raw_limit = os.getenv("CHALLENGE_TABLE_ROW_LIMIT", "100000").strip()
        try:
            self.table_row_limit = max(1000, int(raw_limit))
        except ValueError:
            self.table_row_limit = 100000

    def _iter_zip_files(self, dataset_path: Path) -> list[tuple[str, str]]:
        files: list[tuple[str, str]] = []
        with zipfile.ZipFile(dataset_path, "r") as zip_file:
            for name in zip_file.namelist():
                lower = name.lower()
                if lower.endswith(".csv") or lower.endswith(".json"):
                    text = zip_file.read(name).decode("utf-8", errors="replace")
                    files.append((name, text))
        return files

    def _iter_dir_files(self, dataset_path: Path) -> list[tuple[str, str]]:
        files: list[tuple[str, str]] = []
        for path in dataset_path.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if lower.endswith(".csv") or lower.endswith(".json"):
                files.append((str(path.relative_to(dataset_path)), path.read_text(encoding="utf-8", errors="replace")))
        return files

    def _load_tables(self, dataset_path: Path) -> list[dict[str, Any]]:
        if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
            source_files = self._iter_zip_files(dataset_path)
        elif dataset_path.is_dir():
            source_files = self._iter_dir_files(dataset_path)
        else:
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        tables: list[dict[str, Any]] = []
        for name, text in source_files:
            lower = name.lower()
            if lower.endswith(".csv"):
                reader = csv.DictReader(io.StringIO(text))
                rows: list[dict[str, Any]] = []
                for idx, row in enumerate(reader):
                    if idx >= self.table_row_limit:
                        break
                    rows.append(dict(row))
                if rows:
                    tables.append({"name": name, "rows": rows, "columns": list(rows[0].keys())})
            elif lower.endswith(".json"):
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    rows = [dict(item) for item in data[: self.table_row_limit]]
                    tables.append({"name": name, "rows": rows, "columns": list(rows[0].keys())})

        return tables

    def _id_column_score(self, column_name: str) -> int:
        name = column_name.lower()
        score = 0

        if "transaction" in name:
            score += 8
        if "fraud" in name:
            score += 1
        if name.endswith("_id"):
            score += 4
        if "id" in name:
            score += 2
        if name == "id":
            score += 1
        if "user" in name or "customer" in name:
            score -= 1

        return score

    def _pick_table_and_id_column(self, tables: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        best_table = None
        best_col = ""
        best_score = -1

        for table in tables:
            for column in table["columns"]:
                score = self._id_column_score(column)
                if score > best_score:
                    best_score = score
                    best_table = table
                    best_col = column

        if best_table is None or best_score <= 0:
            raise ValueError(
                "Challenge adapter could not infer a transaction-like ID column. "
                "Update adapter rules after official challenge schema is released."
            )

        return best_table, best_col

    def _find_transactions_table(self, tables: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, str]]:
        for table in tables:
            column_map = {str(col).strip().lower(): str(col) for col in table["columns"]}
            required = {"transaction_id", "sender_id", "recipient_id", "amount", "timestamp"}
            if required.issubset(set(column_map.keys())):
                return table, column_map
        return None, {}

    def _find_table(self, tables: list[dict[str, Any]], keyword: str) -> dict[str, Any] | None:
        needle = keyword.strip().lower()
        for table in tables:
            name = str(table.get("name", "")).lower()
            if needle in name:
                return table
        return None

    def _percentile(self, values: list[float], ratio: float) -> float:
        if not values:
            return 0.0
        bounded = min(max(ratio, 0.0), 1.0)
        idx = int((len(values) - 1) * bounded)
        return values[idx]

    def _robust_z(self, value: float, series: list[float]) -> float:
        if len(series) < 8:
            return 0.0
        center = median(series)
        abs_dev = [abs(item - center) for item in series]
        mad = median(abs_dev)
        scale = mad * 1.4826
        if scale <= 1e-9:
            return 0.0
        return abs(value - center) / scale

    def _parse_timestamp(self, raw: str) -> datetime | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def _tokenize_desc(self, description: str) -> str:
        text = re.sub(r"\s+", " ", description.strip().lower())
        return text

    def _mail_recipient_keyword_scores(self, mails_table: dict[str, Any] | None) -> dict[str, int]:
        if mails_table is None:
            return {}

        risky_keywords = (
            "verify",
            "urgent",
            "password",
            "otp",
            "code",
            "refund",
            "support",
            "security",
            "locked",
            "suspended",
            "confirm",
        )
        recipient_scores: dict[str, int] = defaultdict(int)
        recipient_regex = re.compile(r"\nTo:\s*\"?([^\"<\n]+)", flags=re.IGNORECASE)

        for row in mails_table.get("rows", []):
            if not isinstance(row, dict):
                continue
            content = str(row.get("mail", ""))
            if not content:
                continue
            lowered = content.lower()
            score = 0
            for keyword in risky_keywords:
                if keyword in lowered:
                    score += lowered.count(keyword)
            if score <= 0:
                continue

            match = recipient_regex.search(content)
            if not match:
                continue
            recipient_name = match.group(1).strip().lower()
            if recipient_name:
                recipient_scores[recipient_name] += score

        return dict(recipient_scores)

    def _build_transaction_context(
        self,
        *,
        tx_table: dict[str, Any],
        tx_cols: dict[str, str],
        users_table: dict[str, Any] | None,
        mails_table: dict[str, Any] | None,
        dataset_key: str,
        tables: list[dict[str, Any]],
    ) -> DatasetContext:
        tx_id_col = tx_cols["transaction_id"]
        sender_col = tx_cols["sender_id"]
        recipient_col = tx_cols["recipient_id"]
        amount_col = tx_cols["amount"]
        timestamp_col = tx_cols["timestamp"]

        transaction_rows: list[dict[str, Any]] = []
        for row in tx_table.get("rows", []):
            if not isinstance(row, dict):
                continue
            tx_id = str(row.get(tx_id_col, "")).strip()
            if not tx_id:
                continue

            amount = _to_float(row.get(amount_col))
            sender = str(row.get(sender_col, "")).strip()
            recipient = str(row.get(recipient_col, "")).strip()
            desc = self._tokenize_desc(str(row.get("description", "") or ""))
            tx_type = str(row.get("transaction_type", "") or "").strip().lower()
            payment_method = str(row.get("payment_method", "") or "").strip().lower()
            ts = self._parse_timestamp(str(row.get(timestamp_col, "") or ""))
            balance_after = _to_float(row.get("balance_after"))

            transaction_rows.append(
                {
                    "tx_id": tx_id,
                    "amount": amount,
                    "sender": sender,
                    "recipient": recipient,
                    "desc": desc,
                    "tx_type": tx_type,
                    "payment_method": payment_method,
                    "ts": ts,
                    "hour": ts.hour if ts is not None else -1,
                    "balance_after": balance_after,
                    "raw": row,
                }
            )

        if not transaction_rows:
            raise ValueError("Challenge adapter found no valid transaction rows.")

        users_by_iban: dict[str, dict[str, Any]] = {}
        users_by_name: dict[str, dict[str, Any]] = {}
        if users_table is not None:
            for row in users_table.get("rows", []):
                if not isinstance(row, dict):
                    continue
                iban = str(row.get("iban", "")).strip()
                if iban:
                    users_by_iban[iban] = row
                first_name = str(row.get("first_name", "")).strip()
                last_name = str(row.get("last_name", "")).strip()
                full_name = f"{first_name} {last_name}".strip().lower()
                if full_name:
                    users_by_name[full_name] = row

        recipient_mail_keyword_scores = self._mail_recipient_keyword_scores(mails_table)

        global_amounts = sorted(item["amount"] for item in transaction_rows)
        p90 = self._percentile(global_amounts, 0.90)
        p95 = self._percentile(global_amounts, 0.95)
        p99 = self._percentile(global_amounts, 0.99)

        sender_amounts: dict[str, list[float]] = defaultdict(list)
        recipient_amounts: dict[str, list[float]] = defaultdict(list)
        sender_counts: dict[str, int] = defaultdict(int)
        sender_method_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        sender_type_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        sender_hour_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        description_counts: dict[str, int] = defaultdict(int)

        salary_pair_months: dict[tuple[str, str], set[str]] = defaultdict(set)
        salary_pair_counts: dict[tuple[str, str], int] = defaultdict(int)

        month_tokens = {
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        }

        sorted_for_velocity = sorted(
            transaction_rows,
            key=lambda item: (item["sender"], item["ts"] or datetime.min),
        )
        rapid_tx_ids: set[str] = set()
        previous_by_sender: dict[str, datetime] = {}
        for item in sorted_for_velocity:
            sender = item["sender"]
            current_ts = item["ts"]
            if sender and current_ts is not None and sender in previous_by_sender:
                delta_seconds = (current_ts - previous_by_sender[sender]).total_seconds()
                if 0 <= delta_seconds <= 120:
                    rapid_tx_ids.add(item["tx_id"])
            if sender and current_ts is not None:
                previous_by_sender[sender] = current_ts

        for item in transaction_rows:
            sender = item["sender"]
            recipient = item["recipient"]
            amount = item["amount"]
            tx_type = item["tx_type"]
            method = item["payment_method"]
            hour = item["hour"]
            desc = item["desc"]

            sender_amounts[sender].append(amount)
            recipient_amounts[recipient].append(amount)
            sender_counts[sender] += 1
            if method:
                sender_method_counts[sender][method] += 1
            if tx_type:
                sender_type_counts[sender][tx_type] += 1
            if hour >= 0:
                sender_hour_counts[sender][hour] += 1

            pair_key = (sender, recipient)
            pair_counts[pair_key] += 1
            if desc:
                description_counts[desc] += 1

            if "salary payment" in desc:
                salary_pair_counts[pair_key] += 1
                matched_months = {token for token in month_tokens if token in desc}
                salary_pair_months[pair_key].update(matched_months)

        salary_regular_pairs = {
            pair
            for pair, count in salary_pair_counts.items()
            if count >= 6 and len(salary_pair_months.get(pair, set())) >= 6
        }

        suspicious_desc_keywords = (
            "urgent",
            "verify",
            "otp",
            "password",
            "crypto",
            "bitcoin",
            "gift card",
            "refund",
            "support",
            "security",
            "locked",
            "suspended",
            "wire",
        )
        recurring_benign_markers = (
            "salary payment",
            "insurance premium",
            "monthly",
            "subscription",
            "phone bill",
            "loan payment",
            "donation",
        )

        scored: list[tuple[float, str, str]] = []
        for item in transaction_rows:
            tx_id = item["tx_id"]
            amount = item["amount"]
            sender = item["sender"]
            recipient = item["recipient"]
            tx_type = item["tx_type"]
            method = item["payment_method"]
            hour = item["hour"]
            desc = item["desc"]
            pair_key = (sender, recipient)
            sender_n = sender_counts.get(sender, 0)
            pair_n = pair_counts.get(pair_key, 0)

            sender_z = self._robust_z(amount, sender_amounts.get(sender, []))
            recipient_z = self._robust_z(amount, recipient_amounts.get(recipient, []))

            components = {
                "c_amt": 0.0,
                "c_sender_z": 0.0,
                "c_recipient_z": 0.0,
                "c_pair_rare": 0.0,
                "c_method_rare": 0.0,
                "c_type_rare": 0.0,
                "c_hour_rare": 0.0,
                "c_night": 0.0,
                "c_rapid": 0.0,
                "c_desc_kw": 0.0,
                "c_struct": 0.0,
                "c_balance": 0.0,
                "c_mail": 0.0,
                "c_pair_high": 0.0,
                "c_benign": 0.0,
            }

            if amount >= p99:
                components["c_amt"] += 2.2
            elif amount >= p95:
                components["c_amt"] += 1.4
            elif amount >= p90:
                components["c_amt"] += 0.8

            if sender_z >= 6.0:
                components["c_sender_z"] += 2.0
            elif sender_z >= 4.0:
                components["c_sender_z"] += 1.3
            elif sender_z >= 3.0:
                components["c_sender_z"] += 0.8

            if recipient_z >= 6.0:
                components["c_recipient_z"] += 1.1
            elif recipient_z >= 4.0:
                components["c_recipient_z"] += 0.7

            if sender_n >= 15 and pair_n == 1:
                components["c_pair_rare"] += 0.7

            if sender_n >= 20 and method:
                method_freq = sender_method_counts[sender][method] / float(sender_n)
                if method_freq <= 0.05:
                    components["c_method_rare"] += 0.5

            if sender_n >= 20 and tx_type:
                type_freq = sender_type_counts[sender][tx_type] / float(sender_n)
                if type_freq <= 0.05:
                    components["c_type_rare"] += 0.45

            if sender_n >= 20 and hour >= 0:
                hour_freq = sender_hour_counts[sender][hour] / float(sender_n)
                if hour_freq <= 0.05:
                    components["c_hour_rare"] += 0.4

            if hour in {0, 1, 2, 3, 4}:
                components["c_night"] += 0.25

            if tx_id in rapid_tx_ids:
                components["c_rapid"] += 0.8

            suspicious_hits = sum(1 for keyword in suspicious_desc_keywords if keyword in desc)
            if suspicious_hits > 0:
                components["c_desc_kw"] += min(1.8, 0.6 * suspicious_hits)

            # Extra caution for transfer rows that miss expected fields.
            raw = item["raw"]
            sender_iban = str(raw.get("sender_iban", "") or "").strip()
            recipient_iban = str(raw.get("recipient_iban", "") or "").strip()
            location_text = str(raw.get("location", "") or "").strip()
            if tx_type == "transfer" and (not sender_iban or not recipient_iban):
                components["c_struct"] += 0.7
            if tx_type == "in-person payment" and not location_text:
                components["c_struct"] += 0.35

            if item["balance_after"] > 0 and amount > (item["balance_after"] * 2.0):
                components["c_balance"] += 0.35

            recipient_user = users_by_iban.get(recipient_iban)
            if recipient_user is not None:
                first_name = str(recipient_user.get("first_name", "")).strip()
                last_name = str(recipient_user.get("last_name", "")).strip()
                full_name = f"{first_name} {last_name}".strip().lower()
                if full_name:
                    keyword_score = recipient_mail_keyword_scores.get(full_name, 0)
                    if keyword_score > 0:
                        components["c_mail"] += min(1.1, 0.12 * keyword_score)

            if "salary payment" in desc:
                components["c_benign"] -= 2.6
            if pair_key in salary_regular_pairs:
                components["c_benign"] -= 2.6
            if sender.startswith("EMP") and "salary payment" in desc:
                components["c_benign"] -= 1.2

            if desc and description_counts.get(desc, 0) >= 4 and any(marker in desc for marker in recurring_benign_markers):
                components["c_benign"] -= 1.4

            if pair_n >= 6 and amount >= p95 and not any(marker in desc for marker in recurring_benign_markers):
                components["c_pair_high"] += 0.7

            if "salary payment" in desc:
                desc_tag = "salary"
            elif suspicious_hits > 0:
                desc_tag = "suspicious_keyword"
            elif desc and description_counts.get(desc, 0) >= 4:
                desc_tag = "recurring"
            elif not desc:
                desc_tag = "none"
            else:
                desc_tag = "other"

            risk_hint = sum(components.values())
            line = (
                f"id={tx_id} risk_hint={risk_hint:.4f} amount={amount:.2f} "
                f"sender_tx_count={sender_n} pair_count={pair_n} "
                f"sender_amt_z={sender_z:.2f} recipient_amt_z={recipient_z:.2f} "
                f"hour={hour} rapid_sender={int(tx_id in rapid_tx_ids)} "
                f"type={tx_type or 'none'} method={method or 'none'} desc_tag={desc_tag} "
                f"c_amt={components['c_amt']:.2f} c_sender_z={components['c_sender_z']:.2f} "
                f"c_recipient_z={components['c_recipient_z']:.2f} c_pair_rare={components['c_pair_rare']:.2f} "
                f"c_method_rare={components['c_method_rare']:.2f} c_type_rare={components['c_type_rare']:.2f} "
                f"c_hour_rare={components['c_hour_rare']:.2f} c_night={components['c_night']:.2f} "
                f"c_rapid={components['c_rapid']:.2f} c_desc_kw={components['c_desc_kw']:.2f} "
                f"c_struct={components['c_struct']:.2f} c_balance={components['c_balance']:.2f} "
                f"c_mail={components['c_mail']:.2f} c_pair_high={components['c_pair_high']:.2f} "
                f"c_benign={components['c_benign']:.2f}"
            )
            scored.append((risk_hint, tx_id, line))

        scored.sort(key=lambda item: (-item[0], item[1]))
        shortlisted = [item[1] for item in scored[: self.max_candidate_pool]]
        features_text = "\n".join(item[2] for item in scored[: self.max_candidate_pool])

        table_summaries = []
        for item in tables[:8]:
            table_summaries.append(
                f"table={item['name']} rows={len(item['rows'])} columns={','.join(item['columns'][:20])}"
            )

        summary_text = (
            f"Mode=challenge dataset_key={dataset_key} id_label={self.id_label} entity={self.entity_label}. "
            f"Primary table={tx_table['name']} id_column={tx_id_col}. "
            "Goal: identify suspicious transactions from evolving behavior. "
            "Risk features include amount anomalies, sender/recipient behavior drift, time/method rarity, "
            "rapid sequences, and recurring-legitimate suppression (salary/monthly patterns).\n"
            + "\n".join(table_summaries)
        )

        return DatasetContext(
            dataset_key=dataset_key,
            mode=self.mode,
            id_label=self.id_label,
            entity_label=self.entity_label,
            summary_text=summary_text,
            tool_features_text=features_text,
            candidate_pool=shortlisted,
        )

    def load(self, dataset_path: Path, dataset_key: str) -> DatasetContext:
        tables = self._load_tables(dataset_path)
        if not tables:
            raise ValueError("Challenge adapter found no tabular JSON/CSV data.")

        tx_table, tx_cols = self._find_transactions_table(tables)
        if tx_table is not None and tx_cols:
            users_table = self._find_table(tables, "users")
            mails_table = self._find_table(tables, "mail")
            return self._build_transaction_context(
                tx_table=tx_table,
                tx_cols=tx_cols,
                users_table=users_table,
                mails_table=mails_table,
                dataset_key=dataset_key,
                tables=tables,
            )

        table, id_column = self._pick_table_and_id_column(tables)

        amount_columns = [
            col
            for col in table["columns"]
            if any(key in col.lower() for key in ("amount", "value", "total", "price", "cost", "eur"))
        ]

        count_by_id: dict[str, int] = defaultdict(int)
        amount_sum_by_id: dict[str, float] = defaultdict(float)
        amount_max_by_id: dict[str, float] = defaultdict(float)

        for row in table["rows"]:
            tx_id = str(row.get(id_column, "")).strip()
            if not tx_id:
                continue

            count_by_id[tx_id] += 1
            for column in amount_columns:
                value = _to_float(row.get(column))
                amount_sum_by_id[tx_id] += value
                if value > amount_max_by_id[tx_id]:
                    amount_max_by_id[tx_id] = value

        if not count_by_id:
            raise ValueError("Challenge adapter found no IDs in the inferred ID column.")

        scored: list[tuple[float, str, str]] = []
        for tx_id, count in count_by_id.items():
            amount_max = amount_max_by_id.get(tx_id, 0.0)
            amount_sum = amount_sum_by_id.get(tx_id, 0.0)
            risk_hint = math.log1p(count) + (amount_max * 0.001) + (amount_sum * 0.0001)
            line = (
                f"id={tx_id} risk_hint={risk_hint:.4f} row_count={count} "
                f"amount_max={amount_max:.2f} amount_sum={amount_sum:.2f}"
            )
            scored.append((risk_hint, tx_id, line))

        scored.sort(key=lambda item: (-item[0], item[1]))
        shortlisted = [item[1] for item in scored[: self.max_candidate_pool]]
        features_text = "\n".join(item[2] for item in scored[: self.max_candidate_pool])

        table_summaries = []
        for item in tables[:8]:
            table_summaries.append(
                f"table={item['name']} rows={len(item['rows'])} columns={','.join(item['columns'][:20])}"
            )

        summary_text = (
            f"Mode=challenge dataset_key={dataset_key} id_label={self.id_label} entity={self.entity_label}. "
            f"Inferred primary table={table['name']} id_column={id_column}. "
            "Goal: identify suspicious transactions. "
            "LLM remains the decision brain; deterministic features are support only.\n"
            + "\n".join(table_summaries)
        )

        return DatasetContext(
            dataset_key=dataset_key,
            mode=self.mode,
            id_label=self.id_label,
            entity_label=self.entity_label,
            summary_text=summary_text,
            tool_features_text=features_text,
            candidate_pool=shortlisted,
        )

    def is_valid_id(self, value: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_-]{3,80}", value))
