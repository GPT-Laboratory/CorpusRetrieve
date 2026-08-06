from __future__ import annotations

from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError

from corpusretrieve.config import Settings


class S3ArtifactStore:
    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
        )
        self._ensure_bucket()
        self._ensure_lifecycle(settings.artifact_retention_days)

    def ping(self) -> bool:
        self.client.head_bucket(Bucket=self.bucket)
        return True

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                request = {"Bucket": self.bucket}
                region = self.client.meta.region_name
                if region and region != "us-east-1":
                    request["CreateBucketConfiguration"] = {
                        "LocationConstraint": region
                    }
                self.client.create_bucket(**request)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") not in {
                    "BucketAlreadyExists", "BucketAlreadyOwnedByYou"
                }:
                    raise

    def _ensure_lifecycle(self, retention_days: int) -> None:
        if retention_days <= 0:
            return
        self.client.put_bucket_lifecycle_configuration(
            Bucket=self.bucket,
            LifecycleConfiguration={
                "Rules": [
                    {
                        "ID": "corpusretrieve-job-retention",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "jobs/"},
                        "Expiration": {"Days": retention_days},
                        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
                    }
                ]
            },
        )

    def put_fileobj(self, key: str, stream: BinaryIO, content_type: str | None = None) -> None:
        stream.seek(0)
        extra = {"ContentType": content_type} if content_type else None
        self.client.upload_fileobj(stream, self.bucket, key, ExtraArgs=extra or {})

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        args = {"Bucket": self.bucket, "Key": key, "Body": data}
        if content_type:
            args["ContentType"] = content_type
        self.client.put_object(**args)

    def get_bytes(self, key: str) -> bytes:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                raise FileNotFoundError(key) from exc
            raise

    def stream(self, key: str, chunk_size: int = 1024 * 1024):
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
            yield from body.iter_chunks(chunk_size=chunk_size)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                raise FileNotFoundError(key) from exc
            raise

    def list(self, prefix: str) -> list[dict]:
        paginator = self.client.get_paginator("list_objects_v2")
        output = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                output.append({"key": item["Key"], "size_bytes": item["Size"]})
        return output

    def delete_prefix(self, prefix: str) -> None:
        objects = [{"Key": item["key"]} for item in self.list(prefix)]
        for start in range(0, len(objects), 1000):
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": objects[start:start + 1000], "Quiet": True},
            )

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
