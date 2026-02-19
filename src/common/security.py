"""
Production Security Module
Authentication, authorization, API keys, and security utilities
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
import secrets
from loguru import logger
from functools import wraps
import hashlib


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
security_bearer = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class SecurityManager:
    """Manage authentication and authorization"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.revoked_tokens: set = set()
    
    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a signed JWT with optional custom expiration."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32)  # JWT ID for revocation
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate a JWT, raising HTTPException on failure."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti and jti in self.revoked_tokens:
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            return payload
        
        except JWTError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def revoke_token(self, token: str):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get("jti")
            if jti:
                self.revoked_tokens.add(jti)
                logger.info(f"Token revoked: {jti}")
        except JWTError:
            pass
    
    def generate_api_key(self, name: str, permissions: list) -> str:
        """Create a new API key with the given permission set."""
        api_key = f"omniquant_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "name": name,
            "permissions": permissions,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "usage_count": 0
        }
        
        logger.info(f"API key generated: {name}")
        return api_key
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Look up an API key and update usage stats, or raise 401."""
        if api_key not in self.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Update usage
        self.api_keys[api_key]["last_used"] = datetime.utcnow().isoformat()
        self.api_keys[api_key]["usage_count"] += 1
        
        return self.api_keys[api_key]
    
    def has_permission(self, api_key_data: Dict[str, Any], required_permission: str) -> bool:
        permissions = api_key_data.get("permissions", [])
        return required_permission in permissions or "admin" in permissions


# Global security manager
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    global _security_manager
    if _security_manager is None:
        from src.common.config import get_config
        config = get_config()
        _security_manager = SecurityManager(
            secret_key=config.security.secret_key.get_secret_value()
        )
    return _security_manager


# Dependency injection functions for FastAPI
async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Security(security_bearer)
) -> Dict[str, Any]:
    """Verify JWT bearer token (FastAPI dependency)"""
    security_manager = get_security_manager()
    return security_manager.verify_token(credentials.credentials)


async def verify_api_key_dependency(
    api_key: Optional[str] = Security(api_key_header)
) -> Dict[str, Any]:
    """Verify API key (FastAPI dependency)"""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    security_manager = get_security_manager()
    return security_manager.verify_api_key(api_key)


def require_permission(permission: str):
    """Decorator that checks the caller has the given permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, api_key_data: Dict[str, Any] = None, **kwargs):
            if not api_key_data:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            security_manager = get_security_manager()
            if not security_manager.has_permission(api_key_data, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {permission} required"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self, requests_per_minute: int = 100, burst: int = 20):
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check whether the client is within its per-minute rate limit."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                ts for ts in self.requests[client_id]
                if ts > minute_ago
            ]
        else:
            self.requests[client_id] = []
        
        # Check limits
        request_count = len(self.requests[client_id])
        
        if request_count >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        if client_id not in self.requests:
            return self.requests_per_minute
        return max(0, self.requests_per_minute - len(self.requests[client_id]))


# Global rate limiter
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        from src.common.config import get_config
        config = get_config()
        _rate_limiter = RateLimiter(
            requests_per_minute=config.security.rate_limit_per_minute,
            burst=config.security.rate_limit_burst
        )
    return _rate_limiter


def encrypt_sensitive_data(data: str, key: str) -> str:
    """Encrypt *data* with Fernet symmetric encryption derived from *key*."""
    from cryptography.fernet import Fernet
    import base64
    
    # Generate key from password
    key_bytes = hashlib.sha256(key.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    fernet = Fernet(fernet_key)
    
    encrypted = fernet.encrypt(data.encode())
    return encrypted.decode()


def decrypt_sensitive_data(encrypted_data: str, key: str) -> str:
    from cryptography.fernet import Fernet
    import base64
    
    key_bytes = hashlib.sha256(key.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    fernet = Fernet(fernet_key)
    
    decrypted = fernet.decrypt(encrypted_data.encode())
    return decrypted.decode()


def sanitize_input(input_string: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    # Remove potentially dangerous characters
    dangerous_chars = ["<", ">", "&", '"', "'", ";", "(", ")", "{", "}", "[", "]"]
    sanitized = input_string
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")
    return sanitized.strip()


def audit_log(action: str, user: str, details: Dict[str, Any]):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "user": user,
        "details": details
    }
    logger.info(f"AUDIT: {log_entry}")


if __name__ == "__main__":
    # Test security
    security = SecurityManager(secret_key="test_secret_key_change_in_production")
    
    # Test password hashing
    password = "secure_password_123"
    hashed = security.hash_password(password)
    print(f"Password hashed: {hashed[:20]}...")
    print(f"Verification: {security.verify_password(password, hashed)}")
    
    # Test JWT token
    token = security.create_access_token({"sub": "user123", "role": "trader"})
    print(f"\nJWT Token: {token[:50]}...")
    decoded = security.verify_token(token)
    print(f"Decoded: {decoded}")
    
    # Test API key
    api_key = security.generate_api_key("TestApp", ["trading:read", "trading:execute"])
    print(f"\nAPI Key: {api_key}")
    key_data = security.verify_api_key(api_key)
    print(f"Key Data: {key_data}")
    
    # Test rate limiter
    limiter = RateLimiter(requests_per_minute=5)
    for i in range(7):
        allowed = limiter.is_allowed("client1")
        print(f"Request {i+1}: {'Allowed' if allowed else 'RATE LIMITED'}")
