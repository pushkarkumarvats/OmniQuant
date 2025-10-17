import { View, StyleSheet, ScrollView, Alert } from 'react-native';
import { Text, List, Button, Avatar, ProgressBar, Switch } from 'react-native-paper';
import { useAuthStore } from '@/store/authStore';
import { router } from 'expo-router';
import { useState } from 'react';

export default function ProfileScreen() {
  const { user, profile, signOut } = useAuthStore();
  const [autoSync, setAutoSync] = useState(true);

  const handleSignOut = async () => {
    Alert.alert('Sign Out', 'Are you sure you want to sign out?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Sign Out',
        style: 'destructive',
        onPress: async () => {
          await signOut();
          router.replace('/(auth)/login');
        },
      },
    ]);
  };

  const storageUsed = profile?.storage_used_bytes || 0;
  const storageLimit = profile?.storage_limit_bytes || 104857600; // 100MB
  const storagePercent = (storageUsed / storageLimit) * 100;

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 MB';
    const mb = bytes / 1024 / 1024;
    return `${mb.toFixed(2)} MB`;
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Avatar.Text
          size={80}
          label={profile?.full_name?.charAt(0) || user?.email?.charAt(0) || '?'}
          style={styles.avatar}
        />
        <Text variant="headlineSmall" style={styles.name}>
          {profile?.full_name || 'User'}
        </Text>
        <Text variant="bodyMedium" style={styles.email}>
          {user?.email}
        </Text>
      </View>

      <View style={styles.section}>
        <Text variant="titleMedium" style={styles.sectionTitle}>
          Storage
        </Text>
        <View style={styles.storageCard}>
          <View style={styles.storageHeader}>
            <Text variant="bodyMedium">
              {formatBytes(storageUsed)} / {formatBytes(storageLimit)}
            </Text>
            <Text variant="bodySmall" style={styles.storagePercent}>
              {storagePercent.toFixed(1)}%
            </Text>
          </View>
          <ProgressBar
            progress={storagePercent / 100}
            color={storagePercent > 80 ? '#F44336' : '#1976D2'}
            style={styles.progressBar}
          />
          {storagePercent > 80 && (
            <Text variant="bodySmall" style={styles.storageWarning}>
              Storage almost full. Consider upgrading to Pro.
            </Text>
          )}
        </View>
      </View>

      <View style={styles.section}>
        <Text variant="titleMedium" style={styles.sectionTitle}>
          Settings
        </Text>

        <List.Section>
          <List.Item
            title="Auto-sync"
            description="Automatically sync across devices"
            left={props => <List.Icon {...props} icon="cloud-sync" />}
            right={() => (
              <Switch value={autoSync} onValueChange={setAutoSync} color="#1976D2" />
            )}
          />
          <List.Item
            title="Theme"
            description="Light Mode"
            left={props => <List.Icon {...props} icon="palette" />}
            onPress={() => Alert.alert('Coming Soon', 'Theme settings will be available soon')}
          />
          <List.Item
            title="Notifications"
            description="Manage notification preferences"
            left={props => <List.Icon {...props} icon="bell" />}
            onPress={() =>
              Alert.alert('Coming Soon', 'Notification settings will be available soon')
            }
          />
        </List.Section>
      </View>

      <View style={styles.section}>
        <Text variant="titleMedium" style={styles.sectionTitle}>
          Account
        </Text>

        <List.Section>
          <List.Item
            title="Subscription"
            description="Free Plan"
            left={props => <List.Icon {...props} icon="crown" />}
            right={props => <List.Icon {...props} icon="chevron-right" />}
            onPress={() => Alert.alert('Upgrade', 'Upgrade to Pro for unlimited storage!')}
          />
          <List.Item
            title="Privacy Policy"
            left={props => <List.Icon {...props} icon="shield-check" />}
            right={props => <List.Icon {...props} icon="chevron-right" />}
            onPress={() => Alert.alert('Privacy Policy', 'Privacy policy will open here')}
          />
          <List.Item
            title="Terms of Service"
            left={props => <List.Icon {...props} icon="file-document" />}
            right={props => <List.Icon {...props} icon="chevron-right" />}
            onPress={() => Alert.alert('Terms of Service', 'Terms will open here')}
          />
        </List.Section>
      </View>

      <View style={styles.section}>
        <Button
          mode="outlined"
          onPress={handleSignOut}
          style={styles.signOutButton}
          textColor="#F44336"
        >
          Sign Out
        </Button>
      </View>

      <View style={styles.footer}>
        <Text variant="bodySmall" style={styles.version}>
          BookFlow v1.0.0
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    backgroundColor: '#fff',
    alignItems: 'center',
    paddingTop: 48,
    paddingBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  avatar: {
    marginBottom: 12,
    backgroundColor: '#1976D2',
  },
  name: {
    marginBottom: 4,
    fontWeight: '600',
  },
  email: {
    color: '#757575',
  },
  section: {
    marginTop: 16,
    backgroundColor: '#fff',
    paddingVertical: 16,
  },
  sectionTitle: {
    paddingHorizontal: 16,
    marginBottom: 8,
    fontWeight: '600',
  },
  storageCard: {
    paddingHorizontal: 16,
  },
  storageHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  storagePercent: {
    color: '#757575',
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
  storageWarning: {
    color: '#F44336',
    marginTop: 8,
  },
  signOutButton: {
    marginHorizontal: 16,
    borderColor: '#F44336',
  },
  footer: {
    alignItems: 'center',
    padding: 32,
  },
  version: {
    color: '#9E9E9E',
  },
});
