#!/usr/bin/env bash
# Push the local Bento image to ECR and create (or recreate) an App Runner service.
# Prerequisites: aws CLI configured, Docker running, local image tagged locally (see IMAGE_LOCAL).
#
# App Runner is NOT available in eu-north-1 (Stockholm). ECR + App Runner must use a
# supported region (default: eu-west-1 Ireland). Override: REGION=eu-central-1 bash ...
set -euo pipefail
export AWS_PAGER="${AWS_PAGER:-}"

REGION="${AWS_REGION:-eu-west-1}"
ECR_REPO="${ECR_REPO:-seattle-energy-api}"
SERVICE_NAME="${SERVICE_NAME:-seattle-energy-api}"
# Local Docker image. App Runner needs linux/amd64 — build on Apple Silicon with:
#   bash scripts/rebuild_bento_linux_amd64.sh
# then deploy with the default tag below (or set IMAGE_LOCAL=...).
IMAGE_LOCAL="${IMAGE_LOCAL:-seattle_energy_api:aws-amd64}"
ACCESS_ROLE_NAME="${ACCESS_ROLE_NAME:-AppRunnerECRAccessRole-seattle}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
IMAGE_TAG="${IMAGE_TAG:-$(echo "$IMAGE_LOCAL" | awk -F: '{print $2}')}"
REMOTE_IMAGE="${ECR_URI}:${IMAGE_TAG}"
ACCESS_ROLE_ARN="$(aws iam get-role --role-name "$ACCESS_ROLE_NAME" --query 'Role.Arn' --output text)"

echo "Account=$ACCOUNT Region=$REGION"
echo "Remote image: $REMOTE_IMAGE"

aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" --image-scanning-configuration scanOnPush=true

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
docker tag "$IMAGE_LOCAL" "$REMOTE_IMAGE"
docker push "$REMOTE_IMAGE"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
cat >"$TMP" <<EOF
{
  "ServiceName": "${SERVICE_NAME}",
  "SourceConfiguration": {
    "ImageRepository": {
      "ImageIdentifier": "${REMOTE_IMAGE}",
      "ImageConfiguration": {
        "Port": "3000"
      },
      "ImageRepositoryType": "ECR"
    },
    "AutoDeploymentsEnabled": false,
    "AuthenticationConfiguration": {
      "AccessRoleArn": "${ACCESS_ROLE_ARN}"
    }
  },
  "InstanceConfiguration": {
    "Cpu": "1024",
    "Memory": "2048"
  },
  "HealthCheckConfiguration": {
    "Protocol": "TCP",
    "Interval": 10,
    "Timeout": 5,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 8
  }
}
EOF

if aws apprunner list-services --region "$REGION" --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" --output text | grep -q "arn:"; then
  ARN="$(aws apprunner list-services --region "$REGION" --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn" --output text | awk '{print $1}')"
  echo "Deleting existing App Runner service: $ARN"
  aws apprunner delete-service --region "$REGION" --service-arn "$ARN" >/dev/null
  echo "Waiting ~90s for delete (no AWS waiter for App Runner in all CLI versions)..."
  sleep 90
fi

echo "Creating App Runner service..."
CREATE_OUT="$(aws apprunner create-service --region "$REGION" --cli-input-json "file://${TMP}")"
SVC_ARN="$(echo "$CREATE_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['Service']['ServiceArn'])")"
echo "ServiceArn=$SVC_ARN"
echo "Polling status until RUNNING (deployment often takes 3–8 minutes)..."
for _ in $(seq 1 60); do
  STATUS="$(aws apprunner describe-service --region "$REGION" --service-arn "$SVC_ARN" --query 'Service.Status' --output text)"
  echo "  status=$STATUS"
  if [ "$STATUS" = "RUNNING" ]; then break; fi
  sleep 15
done
URL="$(aws apprunner describe-service --region "$REGION" --service-arn "$SVC_ARN" --query 'Service.ServiceUrl' --output text)"
echo "Service URL: https://${URL}"
echo "Try: curl -sI https://${URL}/"
