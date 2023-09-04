import pytest
from django.core.exceptions import ValidationError

from accounts import forms


# TODO: fix stupid error
# @pytest.mark.usefixtures("")
class TestAcIlEmailSuffixValidator:
    @pytest.mark.parametrize(
        "email",
        ['john.doe@university.ac.il', 'jane.smith@college.ac.il', 'foo.bar@school.ac.il',]
    )
    def test_valid(self, email):
        # The validator should not raise any exception for valid emails
        forms.ac_il_email_validator(email)

    @pytest.mark.parametrize(
        "email",
        ['invalid.email@example.com', 'user@domain.com', 'user@acil', 'user@university.ac.com',]
    )
    def test_invalid(self, email):
        # The validator should raise a ValidationError for invalid emails. Each mail should raise this error
        with pytest.raises(ValidationError):
            forms.ac_il_email_validator(email)
