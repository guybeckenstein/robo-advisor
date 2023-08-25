import pytest
from django.core.exceptions import ValidationError

from accounts import forms


# TODO: fix stupid error
class TestAcIlEmailSuffixValidator:
    def test_valid(self):
        # Valid email addresses
        valid_emails = [
            'john.doe@university.ac.il',
            'jane.smith@college.ac.il',
            'foo.bar@school.ac.il',
        ]

        for email in valid_emails:
            # The validator should not raise any exception for valid emails
            forms.ac_il_email_validator(email)

    def test_invalid(self):
        # Invalid email addresses
        invalid_emails = [
            'invalid.email@example.com',
            'user@domain.com',
            'user@acil',
            'user@university.ac.com',
        ]

        for email in invalid_emails:
            # The validator should raise a ValidationError for invalid emails. Each mail should raise this error
            with pytest.raises(ValidationError):
                forms.ac_il_email_validator(email)
