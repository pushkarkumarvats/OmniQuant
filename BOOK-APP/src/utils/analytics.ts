/**
 * Analytics Utility
 * Tracks user events, screen views, and performance metrics
 * @module utils/analytics
 */

import { logger } from './logger';

type EventName =
  | 'book_opened'
  | 'book_uploaded'
  | 'annotation_created'
  | 'bookmark_created'
  | 'search_performed'
  | 'collection_created'
  | 'pdf_exported'
  | 'user_signed_up'
  | 'user_signed_in'
  | 'subscription_started'
  | string;

interface EventProperties {
  [key: string]: string | number | boolean | null | undefined;
}

interface UserProperties {
  userId?: string;
  email?: string;
  name?: string;
  plan?: 'free' | 'pro' | 'premium';
  totalBooks?: number;
  [key: string]: any;
}

/**
 * Analytics class for tracking user behavior
 */
class Analytics {
  private isEnabled: boolean = true;
  private isDevelopment: boolean;
  private userId?: string;
  private userProperties: UserProperties = {};

  constructor() {
    this.isDevelopment = __DEV__;
  }

  /**
   * Initializes analytics with user information
   * @param userId - User ID
   * @param properties - User properties
   */
  initialize(userId: string, properties?: UserProperties): void {
    this.userId = userId;
    this.userProperties = properties || {};

    if (this.isDevelopment) {
      logger.debug('Analytics initialized', { userId, properties });
    }

    // Initialize analytics services (PostHog, Mixpanel, etc.)
    this.initializeServices();
  }

  /**
   * Tracks an event
   * @param eventName - Name of the event
   * @param properties - Event properties
   */
  trackEvent(eventName: EventName, properties?: EventProperties): void {
    if (!this.isEnabled) return;

    const eventData = {
      event: eventName,
      userId: this.userId,
      timestamp: new Date().toISOString(),
      properties: {
        ...properties,
        platform: this.getPlatform(),
      },
    };

    if (this.isDevelopment) {
      logger.debug(`Event: ${eventName}`, eventData);
    } else {
      // Send to analytics service
      this.sendEvent(eventData);
    }
  }

  /**
   * Tracks a screen view
   * @param screenName - Name of the screen
   * @param properties - Additional properties
   */
  trackScreen(screenName: string, properties?: EventProperties): void {
    this.trackEvent('screen_viewed', {
      screen_name: screenName,
      ...properties,
    });
  }

  /**
   * Tracks a user action
   * @param action - Action name
   * @param category - Category of action
   * @param label - Optional label
   * @param value - Optional numeric value
   */
  trackAction(
    action: string,
    category: string,
    label?: string,
    value?: number
  ): void {
    this.trackEvent('user_action', {
      action,
      category,
      label,
      value,
    });
  }

  /**
   * Tracks an error
   * @param error - Error object or message
   * @param context - Additional context
   */
  trackError(error: Error | string, context?: EventProperties): void {
    const errorMessage = error instanceof Error ? error.message : error;
    const stack = error instanceof Error ? error.stack : undefined;

    this.trackEvent('error_occurred', {
      error_message: errorMessage,
      error_stack: stack,
      ...context,
    });

    logger.error(errorMessage, error);
  }

  /**
   * Tracks a timing/performance metric
   * @param category - Category (e.g., 'PDF', 'API')
   * @param variable - Variable name (e.g., 'load_time')
   * @param time - Time in milliseconds
   * @param label - Optional label
   */
  trackTiming(
    category: string,
    variable: string,
    time: number,
    label?: string
  ): void {
    this.trackEvent('performance_timing', {
      timing_category: category,
      timing_variable: variable,
      timing_value: time,
      timing_label: label,
    });
  }

  /**
   * Identifies a user with properties
   * @param userId - User ID
   * @param properties - User properties
   */
  identifyUser(userId: string, properties?: UserProperties): void {
    this.userId = userId;
    this.userProperties = { ...this.userProperties, ...properties };

    if (this.isDevelopment) {
      logger.debug('User identified', { userId, properties });
    } else {
      // Send to analytics service
      this.identify(userId, this.userProperties);
    }
  }

  /**
   * Updates user properties
   * @param properties - Properties to update
   */
  setUserProperties(properties: UserProperties): void {
    this.userProperties = { ...this.userProperties, ...properties };

    if (!this.isDevelopment) {
      this.identify(this.userId!, this.userProperties);
    }
  }

  /**
   * Tracks revenue/conversion events
   * @param amount - Revenue amount
   * @param currency - Currency code (default: USD)
   * @param productId - Product/plan ID
   */
  trackRevenue(
    amount: number,
    currency: string = 'USD',
    productId?: string
  ): void {
    this.trackEvent('revenue', {
      amount,
      currency,
      product_id: productId,
    });
  }

  /**
   * Resets analytics (e.g., on logout)
   */
  reset(): void {
    this.userId = undefined;
    this.userProperties = {};

    if (this.isDevelopment) {
      logger.debug('Analytics reset');
    }
  }

  /**
   * Enables analytics tracking
   */
  enable(): void {
    this.isEnabled = true;
  }

  /**
   * Disables analytics tracking
   */
  disable(): void {
    this.isEnabled = false;
  }

  /**
   * Gets the current platform
   */
  private getPlatform(): string {
    // Platform detection logic
    return 'web'; // or 'ios', 'android'
  }

  /**
   * Initializes third-party analytics services
   */
  private initializeServices(): void {
    // Initialize PostHog, Mixpanel, Google Analytics, etc.
    // Example:
    // if (typeof window !== 'undefined' && window.posthog) {
    //   window.posthog.identify(this.userId, this.userProperties);
    // }
  }

  /**
   * Sends event to analytics service
   */
  private sendEvent(eventData: any): void {
    // Send to PostHog, Mixpanel, or custom backend
    // Example:
    // if (typeof window !== 'undefined' && window.posthog) {
    //   window.posthog.capture(eventData.event, eventData.properties);
    // }
  }

  /**
   * Identifies user in analytics service
   */
  private identify(userId: string, properties: UserProperties): void {
    // Send to analytics service
    // Example:
    // if (typeof window !== 'undefined' && window.posthog) {
    //   window.posthog.identify(userId, properties);
    // }
  }
}

// Export singleton instance
export const analytics = new Analytics();

// Export convenience functions
export const track = {
  event: (name: EventName, props?: EventProperties) =>
    analytics.trackEvent(name, props),
  screen: (name: string, props?: EventProperties) =>
    analytics.trackScreen(name, props),
  action: (action: string, category: string, label?: string, value?: number) =>
    analytics.trackAction(action, category, label, value),
  error: (error: Error | string, context?: EventProperties) =>
    analytics.trackError(error, context),
  timing: (category: string, variable: string, time: number, label?: string) =>
    analytics.trackTiming(category, variable, time, label),
};

export default analytics;
