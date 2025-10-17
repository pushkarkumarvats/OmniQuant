# OmniQuant Production Deployment Guide

## 🚀 Production Deployment - Complete Guide

This guide covers deploying OmniQuant as **production trading software** with full security, monitoring, and high availability.

---

## ⚠️ Production Readiness Checklist

### Critical Requirements
- [ ] **Environment configured** (`.env` file with all secrets)
- [ ] **Database secured** (PostgreSQL with SSL, strong password)
- [ ] **Redis configured** (Password-protected, persistent)
- [ ] **Broker credentials** (Alpaca/IB API keys validated)
- [ ] **SSL certificates** (HTTPS for API)
- [ ] **Monitoring setup** (Prometheus + Grafana)
- [ ] **Alerting configured** (Slack/Email/PagerDuty)
- [ ] **Backup strategy** (Database + config backups)
- [ ] **Logging configured** (Centralized logging)
- [ ] **Security hardened** (Firewall, rate limiting, API keys)

---

## 1. Environment Configuration

### Create Production `.env` File

```bash
# Environment
ENVIRONMENT=production
DEBUG=false
APP_NAME=OmniQuant
APP_VERSION=2.0.0

# Database (PostgreSQL)
DB_HOST=your-db-host.com
DB_PORT=5432
DB_NAME=omniquant_prod
DB_USER=omniquant_user
DB_PASSWORD=YOUR_STRONG_PASSWORD_HERE  # Change this!
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# Redis
REDIS_HOST=your-redis-host.com
REDIS_PORT=6379
REDIS_PASSWORD=YOUR_REDIS_PASSWORD  # Change this!
REDIS_DB=0
REDIS_MAX_CONNECTIONS=100

# Broker APIs
ALPACA_API_KEY=YOUR_ALPACA_KEY
ALPACA_SECRET_KEY=YOUR_ALPACA_SECRET
ALPACA_BASE_URL=https://api.alpaca.markets  # Live trading URL
POLYGON_API_KEY=YOUR_POLYGON_KEY

# Security (CRITICAL - Generate strong keys!)
SECRET_KEY=YOUR_256_BIT_SECRET_KEY_HERE  # openssl rand -hex 32
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
RATE_LIMIT_PER_MINUTE=1000

# Trading Limits
MAX_POSITION_SIZE=500000.0
MAX_LEVERAGE=2.0
MAX_DRAWDOWN_PCT=0.15
DAILY_LOSS_LIMIT=10000.0
COMMISSION_RATE=0.0002

# Monitoring & Alerting
LOG_LEVEL=INFO
LOG_FORMAT=json
PROMETHEUS_PORT=9090
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
ALERT_EMAIL=alerts@yourcompany.com

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### Generate Secure Keys

```bash
# Generate SECRET_KEY
openssl rand -hex 32

# Generate API keys
python -c "import secrets; print(f'omniquant_{secrets.token_urlsafe(32)}')"
```

---

## 2. Database Setup

### PostgreSQL Installation & Configuration

```bash
# Install PostgreSQL
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql

CREATE DATABASE omniquant_prod;
CREATE USER omniquant_user WITH ENCRYPTED PASSWORD 'YOUR_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE omniquant_prod TO omniquant_user;
ALTER DATABASE omniquant_prod OWNER TO omniquant_user;

# Enable SSL
ALTER SYSTEM SET ssl = on;
SELECT pg_reload_conf();
```

### Run Database Migrations

```bash
# Install Alembic
pip install alembic

# Initialize migrations
alembic init migrations

# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

### Database Backup Strategy

```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/var/backups/omniquant"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/omniquant_$DATE.sql.gz"

mkdir -p $BACKUP_DIR

# Backup with compression
pg_dump -h $DB_HOST -U $DB_USER omniquant_prod | gzip > $BACKUP_FILE

# Keep only last 30 days
find $BACKUP_DIR -name "omniquant_*.sql.gz" -mtime +30 -delete

# Upload to S3 (optional)
aws s3 cp $BACKUP_FILE s3://your-backup-bucket/omniquant/
```

**Cron Schedule** (Daily at 2 AM):
```
0 2 * * * /path/to/backup_database.sh
```

---

## 3. Redis Setup

### Installation & Configuration

```bash
# Install Redis
sudo apt-get install redis-server

# Configure Redis for production
sudo nano /etc/redis/redis.conf
```

**Production Redis Config**:
```conf
# Bind to specific IP
bind 127.0.0.1 your-server-ip

# Require password
requirepass YOUR_STRONG_REDIS_PASSWORD

# Enable persistence
save 900 1
save 300 10
save 60 10000

# AOF persistence
appendonly yes
appendfsync everysec

# Max memory policy
maxmemory 2gb
maxmemory-policy allkeys-lru

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG ""
```

```bash
# Restart Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server
```

---

## 4. Application Deployment

### Option A: Docker Deployment (Recommended)

#### Build Production Image

```bash
# Build optimized image
docker build -t omniquant:2.0.0-prod -f Dockerfile.prod .

# Tag for registry
docker tag omniquant:2.0.0-prod yourdomain.com/omniquant:2.0.0
docker tag omniquant:2.0.0-prod yourdomain.com/omniquant:latest

# Push to registry
docker push yourdomain.com/omniquant:2.0.0
docker push yourdomain.com/omniquant:latest
```

#### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    image: yourdomain.com/omniquant:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

#### Deploy

```bash
# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f api

# Check status
docker-compose -f docker-compose.prod.yml ps
```

### Option B: Kubernetes Deployment

#### Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: omniquant-prod
```

#### Create Secrets

```bash
kubectl create secret generic omniquant-secrets \
  --from-literal=db-password=YOUR_DB_PASSWORD \
  --from-literal=redis-password=YOUR_REDIS_PASSWORD \
  --from-literal=secret-key=YOUR_SECRET_KEY \
  --from-literal=alpaca-api-key=YOUR_ALPACA_KEY \
  --from-literal=alpaca-secret-key=YOUR_ALPACA_SECRET \
  -n omniquant-prod
```

#### Deployment Manifest

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: omniquant-api
  namespace: omniquant-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: omniquant-api
  template:
    metadata:
      labels:
        app: omniquant-api
    spec:
      containers:
      - name: api
        image: yourdomain.com/omniquant:2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: omniquant-secrets
              key: db-password
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: omniquant-secrets
              key: secret-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: omniquant-api-service
  namespace: omniquant-prod
spec:
  selector:
    app: omniquant-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f deployment.yaml

# Check status
kubectl get pods -n omniquant-prod
kubectl get services -n omniquant-prod

# View logs
kubectl logs -f deployment/omniquant-api -n omniquant-prod
```

---

## 5. Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'omniquant'
    static_configs:
      - targets: ['api:9090']
    metrics_path: '/metrics'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - '/etc/prometheus/rules/*.yml'
```

### Alert Rules

```yaml
# alerts.yml
groups:
  - name: trading_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      - alert: DailyLossLimitApproaching
        expr: trading_daily_pnl < -9000
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Daily loss limit approaching"

      - alert: PortfolioDrawdown
        expr: (portfolio_equity - portfolio_peak) / portfolio_peak < -0.15
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Portfolio drawdown exceeded 15%"
```

### Grafana Dashboards

Import pre-built dashboards or create custom ones:

1. Trading Performance Dashboard
2. System Health Dashboard
3. Order Execution Dashboard
4. Risk Metrics Dashboard

---

## 6. Security Hardening

### Firewall Configuration

```bash
# UFW firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API (internal only)
sudo ufw enable

# Restrict database access
sudo ufw allow from your-app-server-ip to any port 5432
```

### SSL/TLS Setup (Nginx)

```nginx
# /etc/nginx/sites-available/omniquant
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
    limit_req zone=api_limit burst=20;
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### API Key Management

```python
# Create API keys for different services
from src.common.security import get_security_manager

security = get_security_manager()

# Trading application
trading_key = security.generate_api_key(
    name="TradingApp",
    permissions=["trading:read", "trading:execute", "portfolio:read"]
)

# Monitoring application (read-only)
monitoring_key = security.generate_api_key(
    name="MonitoringApp",
    permissions=["metrics:read", "health:read"]
)

# Admin application
admin_key = security.generate_api_key(
    name="AdminApp",
    permissions=["admin"]
)

print(f"Trading Key: {trading_key}")
print(f"Monitoring Key: {monitoring_key}")
print(f"Admin Key: {admin_key}")
```

---

## 7. Operational Procedures

### Starting the System

```bash
# 1. Start infrastructure
docker-compose up -d postgres redis

# 2. Run database migrations
alembic upgrade head

# 3. Start application
docker-compose up -d api

# 4. Start monitoring
docker-compose up -d prometheus grafana

# 5. Verify health
curl https://api.yourdomain.com/health
```

### Stopping the System

```bash
# Graceful shutdown
docker-compose stop api

# Wait for open positions to close
# Check via API or dashboard

# Stop infrastructure
docker-compose down
```

### Rolling Updates

```bash
# Pull new image
docker pull yourdomain.com/omniquant:latest

# Update service (zero downtime)
docker-compose up -d --no-deps --build api

# Verify
docker-compose ps
curl https://api.yourdomain.com/health
```

### Disaster Recovery

```bash
# Restore database from backup
gunzip < /var/backups/omniquant/omniquant_20250117_020000.sql.gz | \
  psql -h $DB_HOST -U $DB_USER omniquant_prod

# Restore Redis snapshot
cp /var/backups/redis/dump.rdb /var/lib/redis/

# Restart services
docker-compose restart
```

---

## 8. Monitoring & Alerts

### Key Metrics to Monitor

1. **Trading Metrics**
   - Orders submitted/filled/rejected
   - Trade PnL distribution
   - Position values
   - Daily PnL vs. limits

2. **System Metrics**
   - API latency (p50, p95, p99)
   - Error rates
   - CPU/Memory usage
   - Database connections

3. **Market Data Metrics**
   - Data feed lag
   - Message rates
   - Connection status

### Alert Channels

Configure alerts in `.env`:
```bash
# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK

# Email
ALERT_EMAIL=alerts@yourcompany.com

# PagerDuty (for critical alerts)
PAGERDUTY_API_KEY=YOUR_PAGERDUTY_KEY
```

---

## 9. Compliance & Auditing

### Audit Logging

All trades and critical actions are logged:

```python
from src.common.security import audit_log

audit_log(
    action="order_submitted",
    user="trader_001",
    details={
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "price": 150.00
    }
)
```

### Compliance Checks

- **Pre-trade checks**: Position limits, leverage, concentration
- **Post-trade checks**: Daily loss limits, drawdown limits
- **Regulatory reporting**: Trade logs, position reports

---

## 10. Performance Tuning

### Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_orders_status ON orders(status);

-- Analyze tables
ANALYZE trades;
ANALYZE orders;
ANALYZE positions;
```

### Redis Optimization

```conf
# Increase max clients
maxclients 10000

# Optimize memory
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
```

### Application Tuning

```python
# Use connection pooling
DATABASE_POOL_SIZE = 20
REDIS_MAX_CONNECTIONS = 100

# Enable caching
CACHE_TTL = 300  # 5 minutes

# Async processing
ASYNC_WORKERS = 4
```

---

## 11. Troubleshooting

### Common Issues

**High API Latency**
```bash
# Check database
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

# Check Redis
redis-cli --stat

# Check application logs
docker-compose logs api | grep "ERROR"
```

**Order Rejections**
```bash
# Check broker connection
curl https://api.yourdomain.com/api/v1/broker/status

# Check risk limits
curl https://api.yourdomain.com/api/v1/risk/status
```

**Memory Issues**
```bash
# Check container stats
docker stats

# Restart with more memory
docker-compose up -d --scale api=1 --force-recreate
```

---

## 12. Cost Optimization

### AWS Deployment Costs (Estimated)

| Resource | Specification | Monthly Cost |
|----------|--------------|--------------|
| EC2 (API) | t3.xlarge x2 | $240 |
| RDS (PostgreSQL) | db.t3.large | $140 |
| ElastiCache (Redis) | cache.t3.medium | $50 |
| Load Balancer | ALB | $25 |
| Data Transfer | 1TB/month | $90 |
| Monitoring | CloudWatch | $30 |
| **Total** | | **~$575/month** |

### Cost Saving Tips

1. Use reserved instances (40% savings)
2. Auto-scaling for off-hours
3. S3 for cold storage backups
4. Spot instances for non-critical services

---

## 13. Production Checklist

### Pre-Launch
- [ ] All tests passing (>80% coverage)
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Broker APIs validated (paper trading)
- [ ] Monitoring dashboards configured
- [ ] Alert channels tested
- [ ] Backup/restore tested
- [ ] Documentation complete
- [ ] Team trained on operations

### Launch
- [ ] Deploy to production
- [ ] Verify health checks
- [ ] Monitor metrics closely (first 24h)
- [ ] Start with small positions
- [ ] Gradually increase limits

### Post-Launch
- [ ] Daily monitoring
- [ ] Weekly performance reviews
- [ ] Monthly security updates
- [ ] Quarterly disaster recovery drills
- [ ] Continuous optimization

---

## 🚨 Emergency Contacts

- **On-Call Engineer**: +1-XXX-XXX-XXXX
- **DevOps Team**: devops@yourcompany.com
- **Security Team**: security@yourcompany.com
- **Broker Support**: Alpaca (support@alpaca.markets)

---

## 📚 Additional Resources

- [Kubernetes Production Best Practices](https://kubernetes.io/docs/setup/best-practices/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Redis Production Deployment](https://redis.io/topics/admin)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)

---

**Production Deployment Complete!** 🎉

Your OmniQuant trading system is now running in production with enterprise-grade security, monitoring, and reliability.
