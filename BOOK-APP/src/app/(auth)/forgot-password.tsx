import { useState } from 'react';
import { View, StyleSheet, ScrollView, KeyboardAvoidingView, Platform } from 'react-native';
import { TextInput, Button, Text, HelperText } from 'react-native-paper';
import { Link, router } from 'expo-router';
import { useAuthStore } from '@/store/authStore';

export default function ForgotPasswordScreen() {
  const [email, setEmail] = useState('');
  const [emailError, setEmailError] = useState('');
  const [success, setSuccess] = useState(false);

  const { resetPassword, isLoading, error, clearError } = useAuthStore();

  const validateEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleResetPassword = async () => {
    clearError();
    setEmailError('');
    setSuccess(false);

    if (!validateEmail(email)) {
      setEmailError('Please enter a valid email address');
      return;
    }

    try {
      await resetPassword(email);
      setSuccess(true);
    } catch (err) {
      // Error handled by store
    }
  };

  if (success) {
    return (
      <View style={styles.container}>
        <View style={styles.successContainer}>
          <Text variant="displaySmall" style={styles.successIcon}>
            ✉️
          </Text>
          <Text variant="headlineSmall" style={styles.successTitle}>
            Check Your Email
          </Text>
          <Text variant="bodyLarge" style={styles.successText}>
            We've sent password reset instructions to {email}
          </Text>
          <Button
            mode="contained"
            onPress={() => router.back()}
            style={styles.backButton}
          >
            Back to Sign In
          </Button>
        </View>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text variant="displaySmall" style={styles.logo}>
            📚 BookFlow
          </Text>
          <Text variant="headlineSmall" style={styles.title}>
            Forgot Password?
          </Text>
          <Text variant="bodyLarge" style={styles.subtitle}>
            Enter your email to receive reset instructions
          </Text>
        </View>

        <View style={styles.form}>
          <TextInput
            label="Email"
            value={email}
            onChangeText={text => {
              setEmail(text);
              setEmailError('');
            }}
            keyboardType="email-address"
            autoCapitalize="none"
            autoComplete="email"
            error={!!emailError}
            disabled={isLoading}
            mode="outlined"
            style={styles.input}
          />
          <HelperText type="error" visible={!!emailError}>
            {emailError}
          </HelperText>

          {error && (
            <HelperText type="error" visible style={styles.error}>
              {error}
            </HelperText>
          )}

          <Button
            mode="contained"
            onPress={handleResetPassword}
            loading={isLoading}
            disabled={isLoading}
            style={styles.resetButton}
          >
            Send Reset Link
          </Button>

          <View style={styles.signInContainer}>
            <Text variant="bodyMedium">Remember your password? </Text>
            <Link href="/(auth)/login">
              <Text variant="bodyMedium" style={styles.signInLink}>
                Sign In
              </Text>
            </Link>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 24,
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  logo: {
    marginBottom: 8,
  },
  title: {
    marginBottom: 4,
  },
  subtitle: {
    color: '#757575',
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  form: {
    width: '100%',
    maxWidth: 400,
    alignSelf: 'center',
  },
  input: {
    marginBottom: 8,
  },
  resetButton: {
    marginTop: 8,
    paddingVertical: 6,
  },
  signInContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 24,
  },
  signInLink: {
    color: '#1976D2',
    fontWeight: '600',
  },
  error: {
    marginBottom: 8,
  },
  successContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  successIcon: {
    marginBottom: 16,
  },
  successTitle: {
    marginBottom: 8,
    textAlign: 'center',
  },
  successText: {
    color: '#757575',
    textAlign: 'center',
    marginBottom: 32,
    paddingHorizontal: 32,
  },
  backButton: {
    paddingVertical: 6,
    paddingHorizontal: 24,
  },
});
