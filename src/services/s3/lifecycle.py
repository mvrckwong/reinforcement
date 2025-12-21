"""
S3 Lifecycle Rules Configuration.

Best practice lifecycle rules for RL artifacts:
    - latest checkpoints: Expire after 7 days (recoverable during active training)
    - best checkpoints: Expire after 30 days (keep successful experiments longer)
    - final checkpoints: Transition to Glacier after 30 days (archive indefinitely)
    - logs: Expire after 30 days
    - metadata: Never expires (small, useful for indexing)

Usage:
    from services.s3.lifecycle import configure_lifecycle_rules
    
    configure_lifecycle_rules(client, "artifacts", is_verbose=True)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botocore.client import BaseClient


# Default lifecycle configuration for RL artifacts bucket
DEFAULT_LIFECYCLE_RULES = [
    {
        "ID": "ExpireLatestCheckpoints",
        "Filter": {"Prefix": "checkpoints/latest/"},
        "Status": "Enabled",
        "Expiration": {"Days": 7},
    },
    {
        "ID": "ExpireBestCheckpoints",
        "Filter": {"Prefix": "checkpoints/best/"},
        "Status": "Enabled",
        "Expiration": {"Days": 30},
    },
    {
        "ID": "ArchiveFinalCheckpoints",
        "Filter": {"Prefix": "checkpoints/final/"},
        "Status": "Enabled",
        "Transitions": [
            {"Days": 30, "StorageClass": "GLACIER"},
        ],
    },
    {
        "ID": "ExpireLogs",
        "Filter": {"Prefix": "logs/"},
        "Status": "Enabled",
        "Expiration": {"Days": 30},
    },
]


def configure_lifecycle_rules(
    client: "BaseClient",
    bucket_name: str,
    rules: list[dict] | None = None,
    is_verbose: bool = False,
) -> tuple[bool, str]:
    """Configure S3 lifecycle rules for a bucket.
    
    Note: MinIO supports lifecycle rules but with some limitations.
    This function uses S3-compatible lifecycle configuration.
    
    Args:
        client: S3 client instance.
        bucket_name: Name of the bucket to configure.
        rules: Custom lifecycle rules. Uses DEFAULT_LIFECYCLE_RULES if None.
        is_verbose: Print status messages.
        
    Returns:
        Tuple of (success, error_message).
    """
    rules = rules or DEFAULT_LIFECYCLE_RULES
    
    lifecycle_config = {"Rules": rules}
    
    try:
        client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_config
        )
        
        if is_verbose:
            print(f"✓ Lifecycle rules configured for bucket '{bucket_name}':")
            for rule in rules:
                rule_id = rule.get("ID", "Unknown")
                prefix = rule.get("Filter", {}).get("Prefix", "*")
                if "Expiration" in rule:
                    days = rule["Expiration"].get("Days", "?")
                    print(f"  - {rule_id}: {prefix} → expire after {days} days")
                elif "Transitions" in rule:
                    for t in rule["Transitions"]:
                        storage_class = t.get("StorageClass", "?")
                        days = t.get("Days", "?")
                        print(f"  - {rule_id}: {prefix} → {storage_class} after {days} days")
        
        return (True, "")
    except Exception as e:
        error = str(e)
        if is_verbose:
            print(f"✗ Failed to configure lifecycle rules: {error}")
        return (False, error)


def get_lifecycle_rules(
    client: "BaseClient",
    bucket_name: str,
) -> tuple[list[dict] | None, str]:
    """Get current lifecycle rules for a bucket.
    
    Args:
        client: S3 client instance.
        bucket_name: Name of the bucket.
        
    Returns:
        Tuple of (rules_list or None, error_message).
    """
    try:
        response = client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
        return (response.get("Rules", []), "")
    except client.exceptions.NoSuchLifecycleConfiguration:
        return ([], "")
    except Exception as e:
        return (None, str(e))


def delete_lifecycle_rules(
    client: "BaseClient",
    bucket_name: str,
    is_verbose: bool = False,
) -> tuple[bool, str]:
    """Delete all lifecycle rules from a bucket.
    
    Args:
        client: S3 client instance.
        bucket_name: Name of the bucket.
        is_verbose: Print status messages.
        
    Returns:
        Tuple of (success, error_message).
    """
    try:
        client.delete_bucket_lifecycle(Bucket=bucket_name)
        if is_verbose:
            print(f"✓ Lifecycle rules deleted from bucket '{bucket_name}'")
        return (True, "")
    except Exception as e:
        error = str(e)
        if is_verbose:
            print(f"✗ Failed to delete lifecycle rules: {error}")
        return (False, error)


if __name__ == "__main__":
    pass

