"""
PhishGuard AI — Dataset Generator
Creates training datasets by extracting features from real URL patterns.
Uses curated lists of phishing-style and legitimate-style URLs to generate
realistic feature vectors through the SAME extraction pipeline used at inference.
"""

import pandas as pd
import numpy as np
import os
import re
from urllib.parse import urlparse

from backend.app.config import (
    SUSPICIOUS_TLDS, SUSPICIOUS_KEYWORDS, URL_SHORTENERS, BRAND_DOMAINS
)


# ========================================================================
# Real-world URL patterns for feature extraction
# These are structural templates, NOT hardcoded test cases.
# The model learns from the extracted features, not the URLs themselves.
# ========================================================================

PHISHING_URL_PATTERNS = [
    # Suspicious TLD + keywords
    "http://secure-login-verify.tk/account/confirm",
    "http://account-update-service.ml/verify?user=admin",
    "http://banking-secure-portal.ga/login",
    "http://paypal-security-alert.cf/verify",
    "http://apple-id-confirm.gq/restore",
    "http://microsoft-update.xyz/signin",
    "http://netflix-payment.top/billing/update",
    "http://amazon-order-verify.buzz/confirm",
    "http://facebook-security.club/login/verify",
    "http://google-account.work/security/check",
    "http://instagram-help.info/verify/identity",
    "http://twitter-support.site/account/locked",
    "http://linkedin-verify.online/confirm",
    "http://ebay-dispute.icu/resolve",
    "http://chase-alert.live/security/verify",
    "http://wellsfargo-secure.su/login",
    "http://bankofamerica-update.pw/verify",
    # IP-based URLs
    "http://192.168.1.100/secure/login",
    "http://10.0.0.1/admin/verify",
    "http://172.16.0.50/paypal/login",
    "http://203.0.113.25/bank/verify",
    "http://198.51.100.10/account/update",
    "http://192.0.2.100/microsoft/signin",
    # Long suspicious URLs with many indicators
    "http://secure-bank-login-verify-account.tk/auth/confirm?user=admin&token=abc123",
    "http://paypal-secure-verification-update.ml/login/verify?redirect=home&id=12345",
    "http://apple-id-restore-account-security.ga/confirm/identity?session=xyz",
    "http://microsoft-365-subscription-renew.cf/billing/update?account=user@mail.com",
    "http://google-security-alert-signin.gq/verify/2fa?device=unknown",
    "http://amazon-prime-payment-failed.xyz/update/billing?order=98765",
    "http://netflix-account-suspended-verify.top/restore/access?ref=email",
    # URL shorteners
    "http://bit.ly/3xPhish1",
    "http://tinyurl.com/suspicious-link",
    "http://t.co/fakeAlert99",
    "http://goo.gl/malicious",
    "http://is.gd/phishing1",
    # @ symbol tricks
    "http://www.google.com@evil-site.tk/login",
    "http://paypal.com@phishing-host.ml/verify",
    "http://microsoft.com@attacker.ga/signin",
    # Double slash redirects
    "http://legitimate-looking.com//evil-redirect.tk/login",
    "http://trusted-site.com//phishing-page.ml/verify",
    # Excessive subdomains
    "http://secure.login.verify.account.update.tk/auth",
    "http://banking.security.alert.service.ml/confirm",
    "http://paypal.support.verify.identity.ga/login",
    "http://microsoft.account.update.security.cf/signin",
    # Brand impersonation in subdomains
    "http://paypal.secure-login.tk/verify",
    "http://google.account-alert.ml/signin",
    "http://apple.id-confirm.ga/restore",
    "http://amazon.order-verify.cf/confirm",
    "http://facebook.security-check.gq/login",
    "http://microsoft.365-update.xyz/billing",
    "http://netflix.subscription-verify.top/payment",
    # Many hyphens
    "http://secure-bank-login-verify-now.tk/auth",
    "http://account-security-update-required-urgent.ml/verify",
    "http://paypal-dispute-resolution-center.ga/login",
    # Lots of digits in domain
    "http://secure123login456.tk/verify",
    "http://bank9876account5432.ml/login",
    "http://verify2468confirm1357.ga/auth",
    # Mixed indicators
    "http://192.168.1.50/paypal-login-verify?user=victim@mail.com",
    "http://secure-banking.tk/login//redirect?account=admin&pass=reset",
    "http://bit.ly/2xMalicious",
    "http://account.verify.secure.login.confirm.xyz/auth?token=fake123",
    "http://paypall-security.tk/login",
    "http://g00gle-verify.ml/signin",
    "http://amaz0n-alert.ga/order/confirm",
    "http://micr0soft-update.cf/365/billing",
    "http://faceb00k-security.gq/login/verify",
    "http://netfl1x-payment.xyz/billing/update",
    "http://app1e-id.top/restore/identity",
    "http://tw1tter-support.buzz/account/locked",
    "http://l1nkedin-verify.club/confirm",
    "http://eb4y-dispute.work/resolve",
    "http://ch4se-alert.info/security",
    # HTTP with suspicious paths
    "http://suspicious-domain.tk/wp-admin/login.php",
    "http://phishing-site.ml/cgi-bin/verify.cgi?user=test",
    "http://malicious.ga/.env/credentials",
    "http://evil.cf/admin/password/reset",
    "http://scam.gq/download/malware.exe",
    "http://bad-site.xyz/signin/oauth/callback?code=stolen",
    "http://fake.top/api/v1/auth/token?grant_type=password",
    "http://danger.buzz/update-payment-method",
    "http://hostile.club/verify-your-identity-now",
    "http://threat.work/confirm-account-details",
    # Additional phishing patterns
    "http://free-iphone-giveaway.tk/claim",
    "http://lottery-winner-2024.ml/prize",
    "http://tax-refund-irs.ga/claim-refund",
    "http://crypto-wallet-restore.cf/recovery",
    "http://urgent-security-update.gq/download",
    "http://suspended-account-restore.xyz/verify",
    "http://password-expired-reset.top/update",
    "http://unauthorized-access-alert.buzz/secure",
    "http://verify-identity-immediately.club/confirm",
    "http://click-here-to-confirm.work/auth",
    "http://limited-time-offer-free.info/claim",
    "http://your-account-is-locked.site/unlock",
    "http://unusual-activity-detected.online/verify",
    "http://respond-within-24-hours.icu/confirm",
    "http://final-warning-account.live/restore",
    "http://security-breach-detected.su/protect",
    "http://confirm-your-payment.pw/billing",
    "http://update-expired-card.cc/payment",
    "http://reset-compromised-password.ws/change",
    "http://verify-suspicious-transaction.bid/review",
    "http://claim-your-refund-now.stream/process",
    "http://download-security-patch.download/install",
    "http://free-gift-card-reward.racing/redeem",
    "http://contest-winner-selected.win/claim",
    "http://exclusive-member-offer.party/signup",
    "http://five-star-review-reward.review/earn",
    "http://trading-profit-guaranteed.trade/invest",
    "http://tax-filing-deadline.accountant/submit",
]

LEGITIMATE_URL_PATTERNS = [
    # Well-known sites with HTTPS
    "https://www.google.com",
    "https://www.google.com/search?q=weather",
    "https://www.google.com/maps",
    "https://mail.google.com/inbox",
    "https://drive.google.com/drive/my-drive",
    "https://docs.google.com/document/d/example",
    "https://www.youtube.com",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.microsoft.com",
    "https://www.microsoft.com/en-us/windows",
    "https://outlook.live.com/mail",
    "https://www.office.com",
    "https://github.com",
    "https://github.com/user/repository",
    "https://stackoverflow.com/questions/12345",
    "https://www.amazon.com",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.apple.com",
    "https://www.apple.com/iphone",
    "https://www.facebook.com",
    "https://www.instagram.com",
    "https://www.twitter.com",
    "https://www.linkedin.com",
    "https://www.linkedin.com/in/johndoe",
    "https://www.reddit.com",
    "https://www.reddit.com/r/programming",
    "https://www.wikipedia.org",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.netflix.com",
    "https://www.netflix.com/browse",
    "https://www.paypal.com",
    "https://www.paypal.com/myaccount",
    "https://www.ebay.com",
    "https://www.ebay.com/itm/123456789",
    # E-commerce
    "https://www.walmart.com",
    "https://www.target.com",
    "https://www.bestbuy.com",
    "https://www.etsy.com/listing/123456",
    "https://www.shopify.com",
    # Banking (legitimate)
    "https://www.chase.com",
    "https://www.chase.com/personal/banking",
    "https://www.bankofamerica.com",
    "https://www.wellsfargo.com",
    "https://www.capitalone.com",
    "https://www.citibank.com",
    # News and media
    "https://www.bbc.com/news",
    "https://www.cnn.com",
    "https://www.nytimes.com",
    "https://www.theguardian.com",
    "https://www.reuters.com",
    "https://www.washingtonpost.com",
    # Education
    "https://www.mit.edu",
    "https://www.stanford.edu",
    "https://www.harvard.edu",
    "https://www.coursera.org",
    "https://www.edx.org",
    "https://www.khanacademy.org",
    # Technology
    "https://www.python.org",
    "https://www.python.org/downloads",
    "https://www.npmjs.com",
    "https://www.npmjs.com/package/react",
    "https://developer.mozilla.org/en-US/docs/Web",
    "https://code.visualstudio.com",
    "https://www.docker.com",
    "https://kubernetes.io/docs",
    # Government
    "https://www.usa.gov",
    "https://www.irs.gov",
    "https://www.whitehouse.gov",
    "https://www.nhs.uk",
    # Cloud services
    "https://aws.amazon.com",
    "https://cloud.google.com",
    "https://azure.microsoft.com",
    "https://www.heroku.com",
    "https://www.digitalocean.com",
    # Productivity
    "https://www.notion.so",
    "https://www.slack.com",
    "https://www.zoom.us",
    "https://www.trello.com",
    "https://www.asana.com",
    # Misc legitimate
    "https://www.weather.com",
    "https://www.imdb.com",
    "https://www.spotify.com",
    "https://www.medium.com",
    "https://www.wordpress.com",
    "https://www.blogger.com",
    "https://www.quora.com",
    "https://www.pinterest.com",
    "https://www.tumblr.com",
    "https://www.dropbox.com",
    "https://www.airbnb.com",
    "https://www.uber.com",
    "https://www.lyft.com",
    "https://www.grubhub.com",
    "https://www.doordash.com",
    "https://www.zillow.com",
    "https://www.expedia.com",
    "https://www.booking.com",
    "https://www.tripadvisor.com",
    # Short legitimate
    "https://t.me",
    "https://x.com",
    "https://go.dev",
    "https://rust-lang.org",
    "https://nodejs.org",
    "https://reactjs.org",
    "https://vuejs.org",
    "https://angular.io",
    "https://svelte.dev",
    "https://nextjs.org",
    "https://vercel.com",
    "https://netlify.com",
    "https://www.figma.com",
    "https://www.canva.com",
    # Some with paths and queries (legitimate)
    "https://www.google.com/search?q=phishing+detection&hl=en",
    "https://stackoverflow.com/questions/tagged/python",
    "https://github.com/tensorflow/tensorflow/issues",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://www.amazon.com/gp/cart/view.html",
    "https://www.youtube.com/results?search_query=machine+learning",
    "https://www.linkedin.com/jobs/search/?keywords=developer",
    "https://www.reddit.com/r/cybersecurity/top/?t=month",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://calendar.google.com/calendar/u/0/r",
]


def _extract_features_from_url(url: str) -> dict:
    """Extract features from a URL using the same logic as the detector.
    This ensures training features match inference features exactly."""
    parsed = _safe_parse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    full = url.lower()

    return {
        "url_length": len(url),
        "has_https": 1 if parsed.scheme == "https" else 0,
        "has_ip_address": 1 if _has_ip(hostname) else 0,
        "has_at_symbol": 1 if "@" in url else 0,
        "dot_count": full.count("."),
        "hyphen_count": full.count("-"),
        "subdomain_count": _count_subdomains(hostname),
        "path_length": len(path),
        "query_length": len(parsed.query or ""),
        "has_suspicious_tld": 1 if _has_suspicious_tld(hostname) else 0,
        "suspicious_keyword_count": _count_suspicious_keywords(full),
        "is_shortened": 1 if _is_shortened(hostname) else 0,
        "has_brand_impersonation": 1 if _has_brand_impersonation(hostname, full) else 0,
        "digit_count_in_domain": sum(c.isdigit() for c in hostname),
        "has_double_slash_redirect": 1 if "//" in path else 0,
        "special_char_count": sum(1 for c in full if c in "!#$%^&*()+={}[]|\\:\";< >?,"),
    }


def _safe_parse(url: str):
    if not url.startswith(("http://", "https://", "ftp://")):
        url = "http://" + url
    return urlparse(url)


def _has_ip(hostname: str) -> bool:
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname))


def _count_subdomains(hostname: str) -> int:
    parts = hostname.split(".")
    return max(0, len(parts) - 2)


def _has_suspicious_tld(hostname: str) -> bool:
    return any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS)


def _count_suspicious_keywords(text: str) -> int:
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text)


def _is_shortened(hostname: str) -> bool:
    return any(shortener in hostname for shortener in URL_SHORTENERS)


def _has_brand_impersonation(hostname: str, full_url: str) -> bool:
    for brand, legit_domain in BRAND_DOMAINS.items():
        if brand in hostname and legit_domain not in hostname:
            return True
        parts = hostname.split(".")
        if len(parts) >= 3:
            subdomain_part = ".".join(parts[:-2])
            if brand in subdomain_part and legit_domain not in hostname:
                return True
    return False


def generate_url_dataset(output_path: str = "data/url_dataset.csv") -> pd.DataFrame:
    """
    Generate a URL dataset by extracting features from curated URL patterns.
    Uses the SAME feature extraction pipeline as inference to ensure consistency.
    Labels: 1 = phishing, 0 = legitimate
    """
    rows = []

    # Extract features from phishing URL patterns
    for url in PHISHING_URL_PATTERNS:
        features = _extract_features_from_url(url)
        features["label"] = 1
        rows.append(features)

    # Extract features from legitimate URL patterns
    for url in LEGITIMATE_URL_PATTERNS:
        features = _extract_features_from_url(url)
        features["label"] = 0
        rows.append(features)

    # Add some augmented samples with noise for diversity
    np.random.seed(42)
    augmented = []

    # Augment phishing: slightly vary numeric features
    for url in PHISHING_URL_PATTERNS[:60]:
        features = _extract_features_from_url(url)
        features["url_length"] = max(10, features["url_length"] + np.random.randint(-5, 10))
        features["path_length"] = max(0, features["path_length"] + np.random.randint(-3, 5))
        features["query_length"] = max(0, features["query_length"] + np.random.randint(-2, 5))
        features["label"] = 1
        augmented.append(features)

    # Augment legitimate: slightly vary numeric features
    for url in LEGITIMATE_URL_PATTERNS[:60]:
        features = _extract_features_from_url(url)
        features["url_length"] = max(10, features["url_length"] + np.random.randint(-3, 5))
        features["path_length"] = max(0, features["path_length"] + np.random.randint(-2, 3))
        features["query_length"] = max(0, features["query_length"] + np.random.randint(-1, 3))
        features["label"] = 0
        augmented.append(features)

    rows.extend(augmented)

    dataset = pd.DataFrame(rows)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Remove exact duplicates
    dataset = dataset.drop_duplicates().reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_csv(output_path, index=False)

    n_phishing = dataset["label"].sum()
    n_legitimate = len(dataset) - n_phishing
    print(f"[ML] URL dataset generated: {len(dataset)} samples ({n_phishing} phishing, {n_legitimate} legitimate) -> {output_path}")
    return dataset


def generate_message_dataset(output_path: str = "data/message_dataset.csv") -> pd.DataFrame:
    """
    Generate a demonstration message dataset for TF-IDF + classifier training.
    Labels: 1 = phishing, 0 = legitimate
    """
    phishing_messages = [
        "Your account has been suspended due to unusual activity. Click here to verify your identity immediately: http://secure-login.tk/verify",
        "URGENT: Your bank account will be permanently locked in 24 hours. Confirm your credentials now at http://bank-secure.ml/login",
        "Congratulations! You've won a $1000 gift card. Claim your prize by entering your details: http://free-prize.xyz/claim",
        "Dear customer, we detected unauthorized access to your account. Reset your password immediately: http://account-reset.ga/password",
        "Your payment of $499.99 has been processed. If this wasn't you, click here to cancel: http://payment-verify.cf/cancel",
        "ALERT: Your Netflix subscription expires today. Update your payment method now to avoid interruption: http://netflix-update.tk/billing",
        "Security notice: Someone tried to sign in to your Google account. Verify it's you: http://google-security.ml/verify",
        "Your Apple ID has been disabled. To restore access, confirm your identity here: http://appleid-restore.gq/confirm",
        "FINAL WARNING: Your email storage is full. Click to upgrade immediately or lose your emails: http://email-upgrade.xyz/storage",
        "You have a pending refund of $250. Provide your bank details to receive it: http://refund-process.tk/claim",
        "ACTION REQUIRED: Verify your PayPal account within 12 hours to avoid permanent suspension: http://paypal-verify.ml/secure",
        "Your Instagram account has been flagged for suspicious activity. Verify now: http://instagram-help.ga/verify",
        "Warning: Your credit card ending in 4521 was charged $899. Dispute here: http://card-dispute.cf/report",
        "Dear valued customer, your online banking session has expired. Re-login to continue: http://banking-secure.tk/session",
        "IT Department: Your email password expires today. Update it immediately by clicking this link: http://email-password.ml/update",
        "Tax refund notification: You are eligible for a $1,200 refund. Submit your SSN to process: http://tax-refund.xyz/claim",
        "Your Amazon order #12345 has been cancelled. If this wasn't you, click here: http://amazon-order.ga/verify",
        "URGENT: Your Microsoft 365 subscription payment failed. Update now: http://microsoft-billing.tk/update",
        "Your WhatsApp account will be deleted in 48 hours unless you verify: http://whatsapp-verify.ml/confirm",
        "Exclusive offer! Get 90% off on all products. Limited time only: http://mega-sale.xyz/shop",
        "Your social security number has been compromised. Verify your identity immediately to prevent fraud: http://ssn-verify.tk/protect",
        "Dear user, your account activity requires verification. Enter your OTP here: http://otp-verify.ga/code",
        "NOTICE: Your loan application has been approved for $50,000. Complete verification: http://loan-approved.ml/verify",
        "Your Facebook page has been reported for violations. Appeal now or it will be removed: http://fb-appeal.cf/review",
        "Urgent payment required: Your domain name will expire today. Renew now: http://domain-renew.tk/pay",
        "Security update required for your account. Download the latest security patch: http://security-patch.xyz/download",
        "Your Uber account shows an unpaid trip of $156. Pay now to avoid account suspension: http://uber-pay.ga/settle",
        "Congratulations! You've been selected for a $500 survey reward. Complete it here: http://survey-reward.ml/start",
        "IMPORTANT: Your health insurance coverage will lapse unless you confirm your details: http://insurance-verify.tk/confirm",
        "Alert: Unauthorized login attempt from Russia. Secure your account now: http://account-secure.cf/protect",
        "Dear member, your loyalty points are about to expire. Redeem them now before it's too late: http://points-redeem.tk/claim",
        "Your Spotify account has been compromised. Change your password immediately: http://spotify-secure.ml/reset",
        "FINAL NOTICE: Your electricity will be disconnected unless you pay the outstanding balance: http://electric-pay.ga/bill",
        "You have won a free iPhone 15! Click here to claim your prize before the offer expires: http://free-iphone.xyz/claim",
        "Your LinkedIn account needs verification. Complete the process now to avoid restrictions: http://linkedin-verify.cf/confirm",
    ]

    legitimate_messages = [
        "Dear students, the semester examination schedule has been posted on the college website. Please check the notice board for details.",
        "Reminder: The faculty meeting is scheduled for Thursday at 2 PM in Room 301. Please confirm your attendance.",
        "Your library books are due for return next week. Please visit the library counter during working hours.",
        "The annual sports day event will be held on December 15th. Registration forms are available at the sports office.",
        "Notice: The computer lab will be closed for maintenance on Saturday. Please plan your work accordingly.",
        "Thank you for your purchase at Amazon. Your order #78901 will be delivered by Friday.",
        "Your monthly bank statement for November is ready. Log in to your internet banking portal to view it.",
        "Hi John, the project meeting has been rescheduled to Monday at 10 AM. Let me know if that works.",
        "Welcome to our newsletter! This week's highlights include tips for cybersecurity awareness.",
        "Your flight booking PNR ABC123 has been confirmed. Check-in opens 24 hours before departure.",
        "The quarterly company report is available on the intranet. Please review before the next all-hands meeting.",
        "Reminder: Your dental appointment is on Tuesday at 3:30 PM with Dr. Smith at City Dental Clinic.",
        "The weather forecast shows heavy rain expected this weekend. Stay safe and keep your umbrellas handy.",
        "New course registrations for Spring 2024 open on January 5th. Check the academic calendar for details.",
        "Your gym membership has been renewed for another year. Thank you for being a valued member.",
        "The office holiday party is on December 22nd from 6 PM. RSVP by December 18th.",
        "Your prescription refill is ready for pickup at CVS Pharmacy on Main Street.",
        "The building management reminds all residents to keep the common areas clean.",
        "Your Netflix viewing activity: You watched 5 shows this week. Continue watching your favorites!",
        "Weekly team update: We completed 3 sprints this month. Great work everyone!",
        "Your electricity bill for this month is $85.50. Payment is due by the 15th.",
        "The parent-teacher meeting is scheduled for next Friday. Please arrive by 4 PM.",
        "Congratulations on completing the online certification course! Your certificate is available for download.",
        "Road closure notice: Main Street will be closed for repairs from Dec 10-12. Use alternate routes.",
        "The community garden volunteers meeting is this Saturday at 9 AM at the park pavilion.",
        "Your car service appointment is confirmed for Monday at 10 AM at AutoCare Center.",
        "The local farmers market is open every Sunday from 8 AM to 1 PM at Town Square.",
        "Class notes for today's lecture have been uploaded to the student portal.",
        "The annual charity run registration is now open. Sign up at the community center.",
        "Your package has been delivered to the front door. Thank you for shopping with us.",
        "Hi, the department meeting will be held tomorrow at 10 AM in Seminar Hall 2. Please be present on time.",
        "The hackathon event is scheduled for next weekend. Teams of 3-4 members can register at the CS department.",
        "Your research paper has been accepted for presentation at the conference. Congratulations!",
        "The university cafeteria will have extended hours during exam week. See the schedule posted in the lobby.",
        "Your internship application has been forwarded to the placement cell. They will contact you shortly.",
    ]

    messages = phishing_messages + legitimate_messages
    labels = [1] * len(phishing_messages) + [0] * len(legitimate_messages)

    dataset = pd.DataFrame({"message": messages, "label": labels})
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"[ML] Message dataset generated: {len(dataset)} samples -> {output_path}")
    return dataset


if __name__ == "__main__":
    generate_url_dataset()
    generate_message_dataset()
    print("[ML] All datasets generated successfully.")
