# import pandas as pd
# import numpy as np
# from django.core.management.base import BaseCommand
# from myapp.models import ClaimRecord


# class Command(BaseCommand):
#     help = "Import Safaricom claims data from CSV into ClaimRecord model."

#     def add_arguments(self, parser):
#         parser.add_argument(
#             '--file',
#             type=str,
#             help='Path to the CSV file containing claims data',
#             default='safaricom.csv'
#         )

#     def handle(self, *args, **options):
#         file_path = options['file']
#         self.stdout.write(self.style.NOTICE(f"📂 Reading data from {file_path}..."))

#         # Step 1: Read CSV with encoding fix and avoid dtype issues
#         df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

#         # Step 2: Clean column names
#         df.columns = df.columns.str.strip()
#         print("Cleaned columns:", df.columns.tolist())

#         # Step 3: Convert date columns to proper date format
#         df['CLAIM_PROV_DATE'] = pd.to_datetime(
#             df['CLAIM_PROV_DATE'], format='%d/%m/%Y', errors='coerce'
#         ).dt.date

#         df['DOB'] = pd.to_datetime(
#             df['DOB'], format='%d/%m/%Y', errors='coerce'
#         ).dt.date

#         # Step 4: Clean AMOUNT column
#         df['AMOUNT'] = (
#             df['AMOUNT']
#             .astype(str)  # ensure all entries are strings first
#             .str.replace(',', '', regex=False)  # remove commas
#             .str.strip()  # remove leading/trailing spaces
#         )

#         # Find invalid amount entries before conversion
#         invalid_amounts = df[~df['AMOUNT'].str.match(r'^-?\d+(\.\d+)?$', na=False)]
#         if not invalid_amounts.empty:
#             print("\n⚠ Found invalid AMOUNT values (will be set to 0.0):")
#             print(invalid_amounts[['ADMIT_ID', 'AMOUNT']].head(10))  # show first 10

#         # Replace invalids and convert to float
#         df['AMOUNT'] = (
#             df['AMOUNT']
#             .replace(['-', '', ' - ', ' -   '], np.nan)
#             .astype(float)
#             .fillna(0.0)
#         )

#         # Step 5: Clear old data in ClaimRecord
#         ClaimRecord.objects.all().delete()

#         # Step 6: Create list of ClaimRecord objects
#         records = [
#             ClaimRecord(
#                 admit_id=row['ADMIT_ID'],
#                 icd10_code=row['ICD10_CODE'],
#                 encounter=row['ENCOUNTER'],
#                 service_code=row['CODE'],
#                 quantity=row['QUANTITY'],
#                 amount=row['AMOUNT'],
#                 service_id=row['SERVICE_ID'],
#                 benefit=row['BENEFIT_TYPE'],
#                 benefit_desc=row['BENEFIT_DESC'],
#                 claim_prov_date=row['CLAIM_PROV_DATE'],
#                 claim_pod=row['CLAIM_POOL_NR'],
#                 dob=row['DOB'],
#                 gender=row['Gender'],
#                 dependent_type=row['DependentType'],
#                 ailment=row['Ailment'],
#                 claim_me=row['CLAIM_MEMBER_NUMBER'],
#                 claim_ce=row['CLAIM_CENTRAL_ID'],
#                 claim_in_global=row['CLAIM_INVOICE_NR'],
#                 claim_pa=row['GLOBAL_INVOICE_NR'],
#                 claim_pr=row['CLAIM_PATIENT_FILE_NUMBER'],
#                 prov_name=row['PROV_NAME'],
#                 pol_id=row['POL_ID'],
#                 pol_name=row['POL_NAME'],
#                 cost_center=row['CostCentre']
#             )
#             for _, row in df.iterrows()
#         ]

#         # Step 7: Bulk insert
#         ClaimRecord.objects.bulk_create(records, batch_size=1000)

#         self.stdout.write(self.style.SUCCESS(
#             f"✅ Successfully imported {len(records)} claim records."
#         ))


import pandas as pd
import numpy as np
from decimal import Decimal, InvalidOperation
from django.core.management.base import BaseCommand
from myapp.models import ClaimRecord

class Command(BaseCommand):
    help = "Import Safaricom claims data from CSV into ClaimRecord model."

    def add_arguments(self, parser):
        parser.add_argument(
            "--file",
            type=str,
            help="Path to the CSV file containing claims data",
            default="safaricom.csv",
        )
        parser.add_argument(
            "--replace",
            action="store_true",
            help="Delete all existing ClaimRecord rows before import",
        )
        parser.add_argument(
            "--chunk",
            type=int,
            default=0,
            help="Read CSV in chunks of this size (0 = read all at once)",
        )

    # --- helpers ---
    def _to_decimal(self, s):
        if pd.isna(s):
            return Decimal("0.00")
        try:
            return Decimal(str(s))
        except (InvalidOperation, ValueError):
            return Decimal("0.00")

    def _clean_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize columns
        df.columns = df.columns.str.strip()

        # Dates
        for col, fmt in [("CLAIM_PROV_DATE", "%d/%m/%Y"), ("DOB", "%d/%m/%Y")]:
            if col in df.columns:
                dt = pd.to_datetime(df[col], format=fmt, errors="coerce")
                df[col] = dt.dt.date

        # Amounts
        if "AMOUNT" in df.columns:
            s = (
                df["AMOUNT"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace(["-", "", " - ", " -   "], np.nan)
            )
            # keep as strings; convert to Decimal later row-by-row

            # quick debug: show a few invalid
            invalid = ~s.fillna("").str.match(r"^-?\d+(\.\d+)?$")
            if invalid.any():
                bad = df.loc[invalid, ["ADMIT_ID", "AMOUNT"]].head(10)
                self.stdout.write(self.style.WARNING(f"Found invalid AMOUNT values:\n{bad}"))
            df["AMOUNT"] = s

        return df

    def _rows_to_models(self, df: pd.DataFrame) -> list[ClaimRecord]:
        records = []
        for _, row in df.iterrows():
            records.append(
                ClaimRecord(
                    admit_id=row.get("ADMIT_ID", ""),
                    icd10_code=row.get("ICD10_CODE", ""),
                    encounter=row.get("ENCOUNTER", ""),
                    service_code=row.get("CODE", ""),
                    quantity=int(row.get("QUANTITY", 0) or 0),
                    amount=self._to_decimal(row.get("AMOUNT", "0")),
                    service_id=row.get("SERVICE_ID", ""),
                    benefit=row.get("BENEFIT_TYPE", ""),
                    benefit_desc=row.get("BENEFIT_DESC", ""),
                    claim_prov_date=row.get("CLAIM_PROV_DATE", None),
                    claim_pod=row.get("CLAIM_POOL_NR", ""),
                    dob=row.get("DOB", None),
                    gender=row.get("Gender", ""),
                    dependent_type=row.get("DependentType", ""),
                    ailment=row.get("Ailment", ""),
                    claim_me=row.get("CLAIM_MEMBER_NUMBER", ""),
                    claim_ce=row.get("CLAIM_CENTRAL_ID", ""),
                    claim_in_global=row.get("CLAIM_INVOICE_NR", ""),
                    claim_pa=row.get("GLOBAL_INVOICE_NR", ""),
                    claim_pr=row.get("CLAIM_PATIENT_FILE_NUMBER", ""),
                    prov_name=row.get("PROV_NAME", ""),
                    pol_id=row.get("POL_ID", ""),
                    pol_name=row.get("POL_NAME", ""),
                    cost_center=row.get("CostCentre", ""),
                )
            )
        return records

    def handle(self, *args, **options):
        file_path = options["file"]
        chunk = options["chunk"]
        do_replace = options["replace"]

        self.stdout.write(self.style.NOTICE(f"📂 Reading data from {file_path}..."))

        if do_replace:
            self.stdout.write(self.style.WARNING("Deleting existing ClaimRecord rows..."))
            ClaimRecord.objects.all().delete()

        total = 0
        if chunk and chunk > 0:
            # Chunked import (memory-friendly)
            for i, df in enumerate(pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False, chunksize=chunk)):
                df = self._clean_frame(df)
                batch = self._rows_to_models(df)
                ClaimRecord.objects.bulk_create(batch, batch_size=1000)
                total += len(batch)
                self.stdout.write(self.style.SUCCESS(f"Chunk {i+1}: imported {len(batch)} (total {total})"))
        else:
            # All at once
            df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
            df = self._clean_frame(df)
            batch = self._rows_to_models(df)
            ClaimRecord.objects.bulk_create(batch, batch_size=1000)
            total = len(batch)

        self.stdout.write(self.style.SUCCESS(f"✅ Successfully imported {total} claim records."))
