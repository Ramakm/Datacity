const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface User {
  id: number;
  username: string;
  email: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
}

export interface MessageResponse {
  message: string;
  success: boolean;
}

class AuthApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  private getStoredTokens(): AuthTokens | null {
    if (typeof window === "undefined") return null;
    const tokens = localStorage.getItem("auth_tokens");
    return tokens ? JSON.parse(tokens) : null;
  }

  private storeTokens(tokens: AuthTokens): void {
    if (typeof window === "undefined") return;
    localStorage.setItem("auth_tokens", JSON.stringify(tokens));
  }

  private clearTokens(): void {
    if (typeof window === "undefined") return;
    localStorage.removeItem("auth_tokens");
  }

  getAccessToken(): string | null {
    const tokens = this.getStoredTokens();
    return tokens?.access_token || null;
  }

  isAuthenticated(): boolean {
    return !!this.getAccessToken();
  }

  async register(data: RegisterData): Promise<MessageResponse> {
    const response = await fetch(`${this.baseUrl}/auth/register`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Registration failed");
    }

    return response.json();
  }

  async login(credentials: LoginCredentials): Promise<AuthTokens> {
    const response = await fetch(`${this.baseUrl}/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(credentials),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Login failed");
    }

    const tokens: AuthTokens = await response.json();
    this.storeTokens(tokens);
    return tokens;
  }

  async logout(): Promise<void> {
    const token = this.getAccessToken();
    if (token) {
      try {
        await fetch(`${this.baseUrl}/auth/logout`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
      } catch {
        // Ignore logout errors
      }
    }
    this.clearTokens();
  }

  async refreshTokens(): Promise<AuthTokens | null> {
    const tokens = this.getStoredTokens();
    if (!tokens?.refresh_token) return null;

    try {
      const response = await fetch(`${this.baseUrl}/auth/refresh`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ refresh_token: tokens.refresh_token }),
      });

      if (!response.ok) {
        this.clearTokens();
        return null;
      }

      const newTokens: AuthTokens = await response.json();
      this.storeTokens(newTokens);
      return newTokens;
    } catch {
      this.clearTokens();
      return null;
    }
  }

  async getCurrentUser(): Promise<User | null> {
    const token = this.getAccessToken();
    if (!token) return null;

    try {
      const response = await fetch(`${this.baseUrl}/auth/me`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        if (response.status === 401) {
          // Try to refresh token
          const newTokens = await this.refreshTokens();
          if (newTokens) {
            return this.getCurrentUser();
          }
          return null;
        }
        throw new Error("Failed to get user");
      }

      return response.json();
    } catch {
      return null;
    }
  }
}

export const authApiClient = new AuthApiClient();
